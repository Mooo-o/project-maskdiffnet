import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from geotransformer.modules.transformer.vanilla_transformer import TransformerLayer
from geotransformer.modules.layers import build_dropout_layer

from flash_attn import flash_attn_func
try:
    from apex.normalization import FusedRMSNorm as RMSNorm 
except ModuleNotFoundError:
    print("No fused RMSNorm")
    from rms_norm import RMSNorm

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(MultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)

        self.dropout = build_dropout_layer(dropout)

    def forward(
        self, input_q, input_k, input_v):
        """Vanilla Self-attention forward propagation.

        Args:
            input_q (Tensor): input tensor for query (B, N, C)
            input_k (Tensor): input tensor for key (B, M, C)
            input_v (Tensor): input tensor for value (B, M, C)

        Returns:
            hidden_states: torch.Tensor (B, C, N)
            attention_scores: intermediate values
                'attention_scores': torch.Tensor (B, H, N, M), attention scores before dropout
        """
        # 因为传进来的qkv已经进行多头处理过，这里需要再拼接回去
        # print(f"input_q shape: {input_q.shape}")
        q = rearrange(input_q, 'b n h c -> b n (h c)')
        k = rearrange(input_k, 'b m h c -> b m (h c)')
        v = rearrange(input_v, 'b m h c -> b m (h c)')
        # print(f"q shape: {q.shape}")
        # print(f"self.proj_q weight shape: {self.proj_q.weight.shape}")
        q = rearrange(self.proj_q(q), 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b m (h c) -> b h m c', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b m (h c) -> b h m c', h=self.num_heads)
        
        attention_scores = torch.einsum('bhnc,bhmc->bhnm', q, k) / self.d_model_per_head ** 0.5
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)
        # print(f"v shape: {v.shape}")

        hidden_states = torch.matmul(attention_scores, v)
        # print(f"hidden_states befor shape: {hidden_states.shape}")
        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')
        # print(f"hidden_states after shape: {hidden_states.shape}")
        
        return hidden_states, attention_scores


class MultiheadFlashDiff2(nn.Module):
    """
    DiffAttn implemented with FlashAttention, for packages that does not support different qk/v dimensions
    e.g., flash-attention (https://github.com/Dao-AILab/flash-attention)
    """
    def __init__(
        self,
        embed_dim,
        depth, # current layer index
        num_heads,
        num_kv_heads=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # arg num_heads set to half of baseline Transformer's num_heads
        # for e.g., to compare with a baseline Transformer with 16 heads, pass in num_heads=8 for DIFF Transformer
        self.num_heads = num_heads
        
        # arg num_kv_heads set to half of baseline Transformer's num_kv_heads if use GQA
        # for e.g., to compare with a baseline Transformer with 16 heads and 8 kv_heads, 
        # pass in num_heads=8, num_kv_heads=4 for DIFF Transformer
        # if use MHA, pass in num_kv_heads=None
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.attention = MultiHeadAttention(128, 8, dropout=None) # 1.0 这里用128是因为传进去的qkv的特征维度是128，这里用8是因为diff用的是八个头，需要一致才能重新拼接一下

        # depth means current layer index
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        # self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True) # 0.0
        self.subln = RMSNorm(self.num_heads * 2 * self.head_dim, eps=1e-5, elementwise_affine=True)  # 1.0
    
    def forward(
        self,
        x,
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2, self.head_dim)

        # print(f"q.shape:{q.shape}")
        # print(f"rel_pos[0] shape: {rel_pos[0].shape}, rel_pos[1] shape: {rel_pos[1].shape}")
        # 直接把这两句去掉 q = apply_rotary_emb(q, *rel_pos, interleaved=True) k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        offset = src_len - tgt_len
        q = q.reshape(bsz, tgt_len, self.num_heads, 2, self.head_dim)
        k = k.reshape(bsz, src_len, self.num_kv_heads, 2, self.head_dim)
        
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
        v1, v2 = v[:, :, :, 0], v[:, :, :, 1]

        # 改动
        '''q1 = q1.half()  # 或 .to(torch.float16) k1 = k1.half() v1 = v1.half() q2 = q2.half()  k2 = k2.half() v2 = v2.half()'''# 0.0
        # q1 shape: torch.Size([1, 90, 8, 16]) --> (b, n, h, c)
        # k1 shape: torch.Size([1, 90, 8, 16]) --> (b, n, h, c)
        # v1 shape: torch.Size([1, 90, 8, 16]) --> (b, n, h, c)

        # attn11 = flash_attn_func(q1, k1, v1, causal=True)  # 0.0  attn11 flash shape: torch.Size([1, 90, 8, 16])
        # attn12 = flash_attn_func(q1, k1, v2, causal=True)  # 0.0  attn12 flash shape: torch.Size([1, 90, 8, 16])
        attn11, attn_scores11 = self.attention(q1, k1, v1) # 新增1.0
        attn12, attn_scores12 = self.attention(q1, k1, v2)
        attn1 = torch.cat([attn11, attn12], dim=-1)  # attn flash shape: torch.Size([1, 90, 8, 32])
        # print(f"attn11 shape: {attn11.shape}")
        # print(f"attn12 shape: {attn12.shape}")
        # print(f"attn shape: {attn1.shape}")
        
        # attn21 = flash_attn_func(q2, k2, v1, causal=True) # 0.0
        # attn22 = flash_attn_func(q2, k2, v2, causal=True) # 0.0
        attn21, attn_scores21 = self.attention(q2, k2, v1)
        attn22, attn_scores22 = self.attention(q2, k2, v2)
        attn2 = torch.cat([attn21, attn22], dim=-1)
        
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2

        # print(f"attn befor shape: {attn.shape}")  
        # print(f"self.subln weight shape: {self.subln.weight.shape}")
        
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        # attn = attn.reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)
        
        attn = self.out_proj(attn)
        return attn

# 测试代码
if __name__ == "__main__":
    torch.manual_seed(42)

    # 定义测试参数
    embed_dim = 512
    num_heads = 8
    depth = 3
    seq_len = 16
    batch_size = 64

    # 创建模型
    model = MultiheadFlashDiff2(embed_dim=embed_dim, depth=depth, num_heads=num_heads)
    model = model.cuda()

    # 生成随机输入 ,diff的输入是seq_len表示序列长度（token 数量），embed_dim表示每个token的特征大小
    x = torch.randn(batch_size, seq_len, embed_dim)  # (batch, seq_len, embed_dim)
    #rel_pos = (torch.randn(seq_len, model.head_dim), torch.randn(seq_len, model.head_dim))
    x = x.cuda()
    #rel_pos = (rel_pos[0].cuda(), rel_pos[1].cuda())  # 如果 rel_pos 是 tuple
    print(f"head_dim: {model.head_dim}")
    # print(f"rel_pos[0] shape: {rel_pos[0].shape}, rel_pos[1] shape: {rel_pos[1].shape}")

    # 运行前向传播
    output = model(x)

    # 检查输出形状
    print("Output shape:", output.shape)  # 期望: (batch_size, seq_len, embed_dim)

    # 检查 NaN 或无穷值
    if torch.isnan(output).any() or torch.isinf(output).any():
        print("Warning: Output contains NaN or Inf values!")
    else:
        print("Test passed successfully!")
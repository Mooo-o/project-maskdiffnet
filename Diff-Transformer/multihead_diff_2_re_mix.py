import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from geotransformer.modules.transformer.vanilla_transformer import TransformerLayer
from geotransformer.modules.transformer.rpe_transformer import RPETransformerLayer
from geotransformer.modules.layers import build_dropout_layer

from flash_attn import flash_attn_func
try:
    from apex.normalization import FusedRMSNorm as RMSNorm 
except ModuleNotFoundError:
    print("No fused RMSNorm")
    from rms_norm import RMSNorm

class AttentionOutput(nn.Module):
    def __init__(self, d_model):
        super(AttentionOutput, self).__init__()
        self.expand = nn.Linear(d_model, d_model * 2)
        self.activation = nn.SiLU()
        self.squeeze = nn.Linear(d_model * 2, d_model)
        #self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_states):
        hidden_states = self.expand(input_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.squeeze(hidden_states)
        #hidden_states = self.dropout(hidden_states)
        output_states = self.norm(input_states + hidden_states)
        return output_states

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
    return 0.8 - 0.6 * math.exp(-0.3 * depth) # 


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
        
        self.head_dim = embed_dim // num_heads # // 2
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim * 2 // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim * 2 // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim * 2, embed_dim, bias=False)
        
        self.RPE_selfattention = RPETransformerLayer(256, 8, activation_fn='ReLU'); # 2.0
        # self.attention = TransformerLayer(256, 8, activation_fn='ReLU') # 2.0 这里用128是因为传进去的qkv的特征维度是128，这里用8是因为diff用的是八个头，需要一致才能重新拼接一下

        # depth means current layer index
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        # self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True) # 0.0
        self.subln = RMSNorm(self.num_heads * 2 * self.head_dim, eps=1e-5, elementwise_affine=True)  # 1.0

        self.output = AttentionOutput(embed_dim)  # 新增
    
    def forward(
        self,
        x,
        embedding,
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        #print(f"q.shape before view: {q.shape}")
        #print(f"Expected shape: ({bsz}, {tgt_len}, {2 * self.num_heads}, {self.head_dim})")

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2, self.head_dim)

        # offset = src_len - tgt_len
        q = q.reshape(bsz, tgt_len, self.num_heads, 2, self.head_dim)
        k = k.reshape(bsz, src_len, self.num_kv_heads, 2, self.head_dim)
        
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
        v1, v2 = v[:, :, :, 0], v[:, :, :, 1]

        # q1 shape: torch.Size([1, 90, 8, 16]) --> (b, n, h, c)
        # k1 shape: torch.Size([1, 90, 8, 16]) --> (b, n, h, c)
        # v1 shape: torch.Size([1, 90, 8, 16]) --> (b, n, h, c)
        q1 = rearrange(q1, 'b n h c -> b n (h c)') # 2.0
        k1 = rearrange(k1, 'b m h c -> b m (h c)')
        v1 = rearrange(v1, 'b m h c -> b m (h c)')
        q2 = rearrange(q2, 'b n h c -> b n (h c)')
        k2 = rearrange(k2, 'b m h c -> b m (h c)')
        v2 = rearrange(v2, 'b m h c -> b m (h c)')

        attn11, attn_scores11 = self.RPE_selfattention(10, q1, k1, v1, embedding) # 2.0
        attn12, attn_scores12 = self.RPE_selfattention(10, q1, k1, v2, embedding)
        # attn12, attn_scores12 = self.attention(q1, k1, v2) # 1.0
        attn1 = torch.cat([attn11, attn12], dim=-1)  # attn shape: torch.Size([1, 90, 8, 32])
        
        # attn21, attn_scores21 = self.attention(q2, k2, v1)
        attn21, attn_scores21 = self.RPE_selfattention(10, q2, k2, v1, embedding)
        attn22, attn_scores22 = self.RPE_selfattention(10, q2, k2, v2, embedding)
        attn2 = torch.cat([attn21, attn22], dim=-1)
        
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2
        
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        
        out = self.out_proj(attn)
        
        out = self.output(out)
        
        return out

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
    x = x.cuda()
    print(f"head_dim: {model.head_dim}")

    # 运行前向传播
    output = model(x)

    # 检查输出形状
    print("Output shape:", output.shape)  # 期望: (batch_size, seq_len, embed_dim)

    # 检查 NaN 或无穷值
    if torch.isnan(output).any() or torch.isinf(output).any():
        print("Warning: Output contains NaN or Inf values!")
    else:
        print("Test passed successfully!")
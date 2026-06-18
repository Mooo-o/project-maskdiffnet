r"""Transformer with Relative Positional Embeddings.

Relative positional embedding is further projected in each multi-head attention layer.

The shape of input tensor should be (B, N, C). Implemented with `nn.Linear` and `nn.LayerNorm` (with affine).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from IPython import embed

from geotransformer.modules.layers import build_dropout_layer
from geotransformer.modules.transformer.output_layer import AttentionOutput

class RPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(RPEMultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        self.proj_p = nn.Linear(self.d_model, self.d_model)

        self.dropout = build_dropout_layer(dropout)

        # 动态掩码生成器
        #self.mask_generator = nn.Sequential(
        #    nn.Linear(1, d_model // 2),
        #    nn.ReLU(),
        #    nn.Linear(d_model // 2, 1)
        #)
        self.mask_generator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, i, input_q, input_k, input_v, embed_qk, key_weights=None, key_masks=None, attention_factors=None):
        # (q, k, v, embeddings0, key_masks)
        r"""Scaled Dot-Product Attention with Pre-computed Relative Positional Embedding (forward)

        Args:
            因为这里是自注意力，所以N和M相同
            input_q: torch.Tensor (B, N, C)
            input_k: torch.Tensor (B, M, C)
            input_v: torch.Tensor (B, M, C)
            embed_qk: torch.Tensor (B, N, M, C), relative positional embedding
            key_weights: torch.Tensor (B, M), soft masks for the keys
            key_masks: torch.Tensor (B, M), True if ignored, False if preserved
            attention_factors: torch.Tensor (B, N, M)

        Returns:
            hidden_states: torch.Tensor (B, C, N)
            attention_scores: torch.Tensor (B, H, N, M)
        """
        B, M, _ = input_k.shape

        if i == 0 or i== 4:
            input_q = self.proj_q(input_q)
            input_k = self.proj_k(input_k)
            input_v = self.proj_v(input_v)
        # 重新排布，用来匹配多头
        # print(f"q shape: {q.shape}")
        # print(f"p shape: {embed_qk.shape}")
        # print(f"self.proj_p weight shape: {self.proj_p.weight.shape}")
        q = rearrange(input_q, 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(input_k, 'b m (h c) -> b h m c', h=self.num_heads)
        v = rearrange(input_v, 'b m (h c) -> b h m c', h=self.num_heads)
        p = rearrange(self.proj_p(embed_qk), 'b n m (h c) -> b h n m c', h=self.num_heads)
        
        # einsum 计算 查询 q 和 embed_qk 之间的点积，得到 attention_scores_p
        # q: (B, H, N, C_per_head) 
        # p: (B, H, N, M, C_per_head)
        # attention_scores_p：(B, H, N, M)
        
        # 注意力分数计算
        attention_scores_p = torch.einsum('bhnc,bhnmc->bhnm', q, p)
        attention_scores_e = torch.einsum('bhnc,bhmc->bhnm', q, k)
        attention_scores = (attention_scores_e + attention_scores_p) / self.d_model_per_head ** 0.5
        
        # === 动态掩码生成 ===
        #feature_mean = input_q.mean(dim=2).unsqueeze(-1)  # (B, N, 1)
        #feature_mask = self.mask_generator(feature_mean).squeeze(-1)  # (B, N)
        feature_mask = self.mask_generator(input_q).squeeze(-1)  # (B, N)
        # feature_mask: (B, N) -> 扩展到 (B, H, N, M)，在点对点 attention 上加偏置
        feature_mask = feature_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
        feature_mask = feature_mask.expand(-1, self.num_heads, -1, M)  # (B, H, N, M)
        #print(f"feature_mask shape: {feature_mask}")
        #print(f"attention_scores shape: {attention_scores}")
        
        # 加上mask
        attention_scores = attention_scores + feature_mask

        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        hidden_states = torch.matmul(attention_scores, v)
        
        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')

        return hidden_states, attention_scores

#再加maskattention之前的
'''class RPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(RPEMultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        self.proj_p = nn.Linear(self.d_model, self.d_model)

        self.dropout = build_dropout_layer(dropout)

    def forward(self, i, input_q, input_k, input_v, embed_qk, key_weights=None, key_masks=None, attention_factors=None):
        # (q, k, v, embeddings0, key_masks)
        r"""Scaled Dot-Product Attention with Pre-computed Relative Positional Embedding (forward)

        Args:
            input_q: torch.Tensor (B, N, C)
            input_k: torch.Tensor (B, M, C)
            input_v: torch.Tensor (B, M, C)
            embed_qk: torch.Tensor (B, N, M, C), relative positional embedding
            key_weights: torch.Tensor (B, M), soft masks for the keys
            key_masks: torch.Tensor (B, M), True if ignored, False if preserved
            attention_factors: torch.Tensor (B, N, M)

        Returns:
            hidden_states: torch.Tensor (B, C, N)
            attention_scores: torch.Tensor (B, H, N, M)
        """
        #if i == 0 or i== 4:
        #    input_q = self.proj_q(input_q)
        #    input_k = self.proj_k(input_k)
        #    input_v = self.proj_v(input_v)
            
        # 重新排布，用来匹配多头
        # print(f"q shape: {q.shape}")
        # print(f"p shape: {embed_qk.shape}")
        # print(f"self.proj_p weight shape: {self.proj_p.weight.shape}")
        q = rearrange(input_q, 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(input_k, 'b m (h c) -> b h m c', h=self.num_heads)
        v = rearrange(input_v, 'b m (h c) -> b h m c', h=self.num_heads)
        p = rearrange(self.proj_p(embed_qk), 'b n m (h c) -> b h n m c', h=self.num_heads)
        
        # einsum 计算 查询 q 和 embed_qk 之间的点积，得到 attention_scores_p
        # q: (B, H, N, C_per_head) 
        # p: (B, H, N, M, C_per_head)
        # attention_scores_p：(B, H, N, M)
        
        attention_scores_p = torch.einsum('bhnc,bhnmc->bhnm', q, p)
        attention_scores_e = torch.einsum('bhnc,bhmc->bhnm', q, k)
        attention_scores = (attention_scores_e + attention_scores_p) / self.d_model_per_head ** 0.5
        
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        hidden_states = torch.matmul(attention_scores, v)
        
        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')

        return hidden_states, attention_scores'''


class RPEAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(RPEAttentionLayer, self).__init__()
        self.attention = RPEMultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        i,
        input_states_q,
        memory_states_k,
        memory_states_V,
        position_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ):
        # (q, k, v, embeddings0, key_masks)
        hidden_states, attention_scores = self.attention(
            i,
            input_states_q,
            memory_states_k,
            memory_states_V,
            position_states,
            key_weights=memory_weights,
            key_masks=memory_masks,
            attention_factors=attention_factors,
        )
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states_q)
        return output_states, attention_scores


class RPETransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='ReLU'):
        super(RPETransformerLayer, self).__init__()
        self.attention = RPEAttentionLayer(d_model, num_heads, dropout=dropout)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def forward(
        self,
        i,
        input_states_q,
        memory_states_k,
        memory_states_v,
        embedding_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ):
        # (q, k, v, embeddings0, memory_masks=masks0)
        hidden_states, attention_scores = self.attention(
            i,
            input_states_q,
            memory_states_k,
            memory_states_v,
            embedding_states,
            memory_weights=memory_weights,
            memory_masks=memory_masks,
            attention_factors=attention_factors,
        )
        output_states = self.output(hidden_states)
        return output_states, attention_scores

import torch.nn as nn

from geotransformer.modules.transformer.lrpe_transformer import LRPETransformerLayer
from geotransformer.modules.transformer.pe_transformer import PETransformerLayer
from geotransformer.modules.transformer.rpe_transformer import RPETransformerLayer
from geotransformer.modules.transformer.vanilla_transformer import TransformerLayer

import sys
sys.path.append("/root/autodl-tmp/GeoTransformer/Diff-Transformer")

import torch
from multihead_diff_2_re_mix import MultiheadFlashDiff2 #这里也要改变

try:
    from apex.normalization import FusedRMSNorm as RMSNorm 
except ModuleNotFoundError:
    print("No fused RMSNorm")
    from rms_norm import RMSNorm


def _check_block_type(block):
    if block not in ['self', 'cross', 'diff']:
        raise ValueError('Unsupported block type "{}".'.format(block))


class ResidualGatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.ReLU(),  # or Tanh()
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, feats, new_feats):
        feats = self.norm(feats)
        new_feats = self.norm(new_feats)
        concat = torch.cat([feats, new_feats], dim=-1)

        if torch.isnan(concat).any():
            print("Concat has NaNs")

        gate = self.gate(concat)

        if torch.isnan(gate).any():
            print("Gate has NaNs!")
        
        gate = torch.clamp(gate, 1e-4, 1 - 1e-4)
        fused = gate * new_feats + (1 - gate) * feats
        return feats + fused




class VanillaConditionalTransformer(nn.Module):
    def __init__(self, blocks, d_model, num_heads, dropout=None, activation_fn='ReLU', return_attention_scores=False):
        super(VanillaConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1


class PEConditionalTransformer(nn.Module):
    def __init__(self, blocks, d_model, num_heads, dropout=None, activation_fn='ReLU', return_attention_scores=False):
        super(PEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(PETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, embeddings0, embeddings1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, embeddings0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, embeddings1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1


class RPEConditionalTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        depth,
        blocks,
        d_model,
        num_heads,
        dropout=None,
        activation_fn='ReLU',
        return_attention_scores=False,
        parallel=False,
    ):
        super(RPEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for i, block in enumerate(self.blocks):
            _check_block_type(block)
            if block == 'self':
                 layers.append(RPETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn)) # 自注意力
            elif block == 'cross':
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn)) # 交叉注意力
            else :
                if i == 2:
                    layers.append(MultiheadFlashDiff2(embed_dim=embed_dim, depth=2, num_heads=num_heads * 2))# diff attention
                elif i == 6:
                    layers.append(MultiheadFlashDiff2(embed_dim=embed_dim, depth=depth, num_heads=num_heads * 2))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores
        self.parallel = parallel
        
        self.w0 = nn.Parameter(torch.tensor(0.5))  # fusion特征融合系数
        self.w1 = nn.Parameter(torch.tensor(0.5))
        self.w2 = nn.Parameter(torch.tensor(1.0))  
        self.w3 = nn.Parameter(torch.tensor(1.0))

        #self.residual_gating_1 = ResidualGatedFusion(d_model)
        #self.residual_gating_2 = ResidualGatedFusion(d_model)


    def forward(self, feats0, feats1, embeddings0, embeddings1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](i, feats0, feats0, feats0, embeddings0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](i, feats1, feats1, feats1, embeddings1, memory_masks=masks1)
            elif block == 'cross':
                if i == 1 or i == 5:
                    new_feats0, scores0 = self.layers[i](feats0, feats1, feats1, memory_masks=masks1)
                    new_feats1, scores1 = self.layers[i](feats1, feats0, feats0, memory_masks=masks0)
                    feats0 = new_feats0.clone()
                    feats1 = new_feats1.clone()
                else:
                    feats0, scores0 = self.layers[i](feats0, feats1, feats1, memory_masks=masks1)
                    feats1, scores1 = self.layers[i](feats1, feats0, feats0, memory_masks=masks0)
            else: # diff
                feats0 = self.layers[i](feats0.clone(), embeddings0)
                feats1 = self.layers[i](feats1.clone(), embeddings1)
                if i == 2:
                    feats0 = new_feats0 + feats0 * torch.sigmoid(self.w0)
                    feats1 = new_feats1 + feats1 * torch.sigmoid(self.w1)
                    #feats0 = self.residual_gating_1(new_feats0, feats0)
                    #feats1 = self.residual_gating_1(new_feats1, feats1)
                if i == 6:
                    feats0 = new_feats0 + feats0 * torch.sigmoid(self.w2)
                    feats1 = new_feats1 + feats1 * torch.sigmoid(self.w3)
                    # feats0 = self.residual_gating_2(new_feats0, feats0)
                    # feats1 = self.residual_gating_2(new_feats1, feats1)
                    #print(f"[Debug] w0: {self.w0.item()} w1: {self.w1.item()} w2: {self.w2.item()} w3: {self.w3.item()}") # 1.0 1.0 0.847 0.1
                    
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1


class LRPEConditionalTransformer(nn.Module):
    def __init__(
        self,
        blocks,
        d_model,
        num_heads,
        num_embeddings,
        dropout=None,
        activation_fn='ReLU',
        return_attention_scores=False,
    ):
        super(LRPEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(
                    LRPETransformerLayer(
                        d_model, num_heads, num_embeddings, dropout=dropout, activation_fn=activation_fn
                    )
                )
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, emb_indices0, emb_indices1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, emb_indices0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, emb_indices1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1

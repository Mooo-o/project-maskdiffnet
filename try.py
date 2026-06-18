import sys
sys.path.append("/root/autodl-tmp/GeoTransformer/Diff-Transformer")

import torch
from multihead_flashdiff_2_re import MultiheadFlashDiff2

try:
    from apex.normalization import FusedRMSNorm as RMSNorm 
except ModuleNotFoundError:
    print("No fused RMSNorm")
    from rms_norm import RMSNorm

# 检查 CUDA 是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"

# 创建模型
embed_dim = 256
num_heads = 16
depth = 3
seq_len = 16
batch_size = 64

model = MultiheadFlashDiff2(embed_dim=embed_dim, depth=depth, num_heads=num_heads).to(device)

# 生成输入
x = torch.randn(batch_size, seq_len, embed_dim).to(device)

# 前向传播
output = model(x)
print("Output shape:", output.shape)

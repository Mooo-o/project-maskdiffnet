# GeoTransformer / MaskDiffNet Project

This repository implements **GeoTransformer / MaskDiffNet** for 3D point cloud registration tasks.

---

## 🚀 Overview

This project focuses on **robust 3D point cloud registration**, especially under:
- Low overlap scenarios (3DLoMatch)
- Noise and outliers
- Large scene variations

---

## 📦 Environment Setup

follow the repo https://github.com/qinzheng93/GeoTransformer

```bash
# It is recommended to create a new environment
conda create -n geotransformer python==3.8
conda activate geotransformer

# [Optional] If you are using CUDA 11.0 or newer, please install `torch==1.7.1+cu110`
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# Install packages and other dependencies
pip install -r requirements.txt
python setup.py build develop

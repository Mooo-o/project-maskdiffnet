# GeoTransformer - MaskDiffNet Project

This repository implements **GeoTransformer - MaskDiffNet** for 3D point cloud registration tasks. **GeoTransformer** is the baseline.

---

## 🚀 Overview

This project focuses on **robust 3D point cloud registration**, especially under:
- Low overlap scenarios (3DLoMatch)
- Noise and outliers
- Large scene variations

---

## 📦 Environment Setup

Follow the repo https://github.com/qinzheng93/GeoTransformer and the repo https://github.com/microsoft/unilm/tree/master/Diff-Transformer.

All required dependencies are listed in `list.md` for reference.

```bash
# It is recommended to create a new environment
conda create -n geotransformer python==3.8
conda activate geotransformer

# [Optional] If you are using CUDA 11.0 or newer, please install `torch==1.7.1+cu110`
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# Install packages and other dependencies
pip install -r requirements.txt
python setup.py build develop
```

---

## 📊 Datasets
### 3DMatch
The dataset can be downloaded from [PREDATOR](https://github.com/prs-eth/OverlapPredator). The data should be organized as follows:

```text
--data--3DMatch--metadata
              |--data--train--7-scenes-chess--cloud_bin_0.pth
                    |      |               |--...
                    |      |--...
                    |--test--7-scenes-redkitchen--cloud_bin_0.pth
                          |                    |--...
                          |--...
```
### ModelNet

Download the [data](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) and run `data/ModelNet/split_data.py` to generate the data. The data should be organized as follows:

```text
--data--ModelNet--modelnet_ply_hdf5_2048--...
               |--train.pkl
               |--val.pkl
               |--test.pkl
```
### Kitti odometry

Download the data from the [Kitti official website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) into `data/Kitti` and run `data/Kitti/downsample_pcd.py` to generate the data. The data should be organized as follows:

```text
--data--Kitti--metadata
            |--sequences--00--velodyne--000000.bin
            |              |         |--...
            |              |...
            |--downsampled--00--000000.npy
                         |   |--...
                         |--...
```

---

## 🏋️ train

```bash
# 3DMatch/3DLoMatch
conda activate geotransformer
cd experiments/geo.3dmatch.scdc2.SiLU.maskatten.warmup.loss
CUDA_VISIBLE_DEVICES=0 python trainval.py

# Modelnet
conda activate geotransformer
cd experiments/geo.modelnet.selfcrossdiffcross2.SiLU.maskatten.warmup
CUDA_VISIBLE_DEVICES=0 python trainval.py

# 在kitti上的改动之前因为云服务器过期没有保存但是修改原理和在3dmatch和Modelnet一样，如果需要验证kitti数据集上的效果需自行修改
```

## 🧪 Testing / Evaluation

### 3DMatch/3DLoMatch
```bash
# 3DMatch,因为是要用自己训出来的权重来测试所以--snapshot参数需要改成对应权重的路径，下面仅为示范
CUDA_VISIBLE_DEVICES=0 python test.py --snapshot=../../output/geo.3dmatch.scdc2.SiLU.maskatten.warmup.loss/snapshots/epoch-39.pth.tar --benchmark=3DMatch
CUDA_VISIBLE_DEVICES=0 eval.py --test_epoch=39 --benchmark=3DMatch --method=lgr
# 3DLoMatch
CUDA_VISIBLE_DEVICES=0 python test.py --snapshot=../../output/geo.3dmatch.scdc2.SiLU.maskatten.warmup.loss/snapshots/epoch-39.pth.tar --benchmark=3DLoMatch
CUDA_VISIBLE_DEVICES=0 eval.py --test_epoch=39 --benchmark=3DLoMatch --method=lgr
```
EPOCH is the epoch id.

### Modelnet
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --test_iter=ITER
eg. CUDA_VISIBLE_DEVICES=0 test.py --test_iter=400000
    CUDA_VISIBLE_DEVICES=0 test.py --test_iter=390000
```
ITER is the iteration id.

### Kitti
在kitti上的改动之前因为云服务器过期没有保存但是修改原理和在3dmatch和Modelnet一样，如果需要验证kitti数据集上的效果需自行修改

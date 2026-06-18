import argparse
import os.path as osp
import time

import numpy as np

from geotransformer.engine import SingleTester
from geotransformer.utils.torch import release_cuda
from geotransformer.utils.common import ensure_dir, get_log_string

from dataset import test_data_loader
from config import make_cfg
from model import create_model
from loss import Evaluator

import open3d as o3d
import matplotlib.pyplot as plt
import torch

import os
import torch
import open3d as o3d

def save_high_attention_patches(ref_points_c, ref_node_knn_points, attention_scores, layer_id, iteration, topk=30):
    """
    保存 attention 分数最高的前 topk 个 ref 超点及其邻域点
    """
    # 创建保存路径
    save_root = os.path.join('visualization', f'{iteration:06d}')
    os.makedirs(save_root, exist_ok=True)
    print(f"[Debug] attention_scores shape: {attention_scores[layer_id][0].shape}, topk: {topk}")

    save_path = os.path.join(save_root, f'{iteration:06d}_layer{layer_id}_top{topk:02d}.ply')

    # 获取对应层的 cross-attention 分数（形状：(1, head, N, M)）
    attn = attention_scores[layer_id][0].squeeze(0).mean(0)  # shape: (N, M) -> N 是 ref 超点数量
    print(f"[Debug] attn shape: {attn.shape}, topk: {topk}")
    topk_indices = torch.topk(attn.max(dim=1).values, k=topk).indices  # shape: (topk,)
    print(f"[Debug] topk_indices.max: {topk_indices.max()}, ref_points_c.shape: {ref_points_c.shape}")
    print(f"[Debug] ref_node_knn_indices.shape: {ref_node_knn_points.shape}")
    '''selected_superpoints = ref_points_c[topk_indices]  # (topk, 3)
    selected_patches = ref_node_knn_points[topk_indices]  # (topk, K, 3)

    all_points = []
    all_colors = []

    for sp, patch in zip(selected_superpoints, selected_patches):
        all_points.append(sp.unsqueeze(0))           # 超点本身
        all_colors.append(torch.tensor([[1, 0, 0]]))  # 红色

        all_points.append(patch)                      # 周围点
        all_colors.append(torch.tensor([[0.5, 0.5, 0.5]]).expand(patch.shape[0], -1))  # 灰色

    all_points = torch.cat(all_points, dim=0).cpu().numpy()
    all_colors = torch.cat(all_colors, dim=0).cpu().numpy()

    # 写入 ply
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    o3d.io.write_point_cloud(save_path, pcd)
    print(f"[Save] Saved high-attention patches to: {save_path}")'''


def save_attention_ply(points, attention_map, save_path, topk=1000):
    # attention_map: (N, M)
    scores = attention_map.max(dim=1).values  # 每个 ref 点对应的最大注意力值
    topk_indices = torch.topk(scores, topk).indices
    topk_points = points[topk_indices].cpu().numpy()

    # 创建 Open3D 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(topk_points)

    # 可选：上色（比如根据注意力强度）
    normalized_scores = scores[topk_indices].cpu().numpy()
    normalized_scores = (normalized_scores - normalized_scores.min()) / (normalized_scores.ptp() + 1e-8)
    colors = plt.get_cmap("Reds")(normalized_scores)[:, :3]  # RGB
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(save_path, pcd)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=['3DMatch', '3DLoMatch', 'val'], help='test benchmark')
    return parser


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg, parser=make_parser())
        
        # dataloader
        start_time = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg, self.args.benchmark)
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)
        message = f'Calibrate neighbors: {neighbor_limits}.'
        self.logger.info(message)
        self.register_loader(data_loader)

        # model
        model = create_model(cfg).cuda()
        self.register_model(model)

        # evaluator
        self.evaluator = Evaluator(cfg).cuda()

        # preparation
        self.output_dir = osp.join(cfg.feature_dir, self.args.benchmark)
        ensure_dir(self.output_dir)

    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.evaluator(output_dict, data_dict)
        return result_dict

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        scene_name = data_dict['scene_name']
        ref_frame = data_dict['ref_frame']
        src_frame = data_dict['src_frame']
        message = f'{scene_name}, id0: {ref_frame}, id1: {src_frame}'
        message += ', ' + get_log_string(result_dict=result_dict)
        message += ', nCorr: {}'.format(output_dict['corr_scores'].shape[0])
        return message

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        scene_name = data_dict['scene_name']
        ref_id = data_dict['ref_frame']
        src_id = data_dict['src_frame']

        ensure_dir(osp.join(self.output_dir, scene_name))
        file_name = osp.join(self.output_dir, scene_name, f'{ref_id}_{src_id}.npz')
        np.savez_compressed(
            file_name,
            ref_points=release_cuda(output_dict['ref_points']),
            src_points=release_cuda(output_dict['src_points']),
            ref_points_f=release_cuda(output_dict['ref_points_f']),
            src_points_f=release_cuda(output_dict['src_points_f']),
            ref_points_c=release_cuda(output_dict['ref_points_c']),
            src_points_c=release_cuda(output_dict['src_points_c']),
            ref_feats_c=release_cuda(output_dict['ref_feats_c']),
            src_feats_c=release_cuda(output_dict['src_feats_c']),
            ref_node_corr_indices=release_cuda(output_dict['ref_node_corr_indices']),
            src_node_corr_indices=release_cuda(output_dict['src_node_corr_indices']),
            ref_corr_points=release_cuda(output_dict['ref_corr_points']),
            src_corr_points=release_cuda(output_dict['src_corr_points']),
            corr_scores=release_cuda(output_dict['corr_scores']),
            gt_node_corr_indices=release_cuda(output_dict['gt_node_corr_indices']),
            gt_node_corr_overlaps=release_cuda(output_dict['gt_node_corr_overlaps']),
            estimated_transform=release_cuda(output_dict['estimated_transform']),
            transform=release_cuda(data_dict['transform']),
            overlap=data_dict['overlap'],
            #attention_scores=release_cuda(output_dict['attention_scores']),   # 新增
        )
        #print("output_dict keys:", output_dict.keys())
        '''if iteration < 12:
            save_high_attention_patches(
                ref_points_c=output_dict['ref_points_c'],
                ref_node_knn_points=output_dict['ref_node_corr_knn_points'],  # 注意用的是 coarse 匹配点的 patch
                attention_scores=output_dict['attention_scores'],
                layer_id=6,
                iteration=iteration,
                topk=30
            )

            save_high_attention_patches(
                ref_points_c=output_dict['ref_points_c'],
                ref_node_knn_points=output_dict['ref_node_corr_knn_points'],
                attention_scores=output_dict['attention_scores'],
                layer_id=7,
                iteration=iteration,
                topk=30
            )'''

        
        '''if iteration < 20:
            ref_points = output_dict['ref_points']  # shape: (N, 3)
            attention_scores = output_dict['attention_scores']  # list of [layer][0/1]

            for layer_idx in [5, 6]:  # 第五、第六层
                attention = attention_scores[layer_idx][0]  # ref -> src
                attention_mean = attention.mean(dim=1).squeeze(0)  # shape: (N, M)

                ply_name = f'{ref_id}_{src_id}_layer{layer_idx+1}_attention.ply'
                ply_path = osp.join(self.output_dir, scene_name, ply_name)
                save_attention_ply(ref_points, attention_mean, ply_path, topk=1000)'''


        #attention_scores = release_cuda(output_dict['attention_scores'])

        #print(f'[Debug] attention_scores shape: {attention_scores.shape}')


def main():
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()


if __name__ == '__main__':
    main()

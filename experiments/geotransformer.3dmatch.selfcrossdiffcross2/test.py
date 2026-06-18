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


def save_high_attention_superpoints_and_patches(output_dict, layer_ids=[5, 6], iteration=None, topk_ratio=0.3):
    # 提取必要的数据
    ref_points_c_y = output_dict['ref_points_c_y'].cpu().numpy()  # (N, 3)
    print(f'[Debug] ref_points_c_y shape: {ref_points_c_y.shape}')  # (N, 3)
    ref_node_knn_points = output_dict['ref_node_knn_points'].cpu().numpy()  # (N, K, 3)
    attention_scores = output_dict['attention_scores']  # list of len L, each: (1, H, N, M)

    save_dir = os.path.join('visualization', f'{iteration:06d}')
    os.makedirs(save_dir, exist_ok=True)
    
    for lid in layer_ids:
        attn = attention_scores[lid][0].squeeze(0)  # (H, N, M)，第lid层
        print(f'[Debug] attention_scores shape: {attn.shape}')  # (H, N, M)
        attn_score_per_node = attn.mean(dim=(0, 2))  # (N,) —— 对 head 和 src 点求平均，得到每个 ref 超点的注意力强度
        print(f'[Debug] attn_score_per_node shape: {attn_score_per_node.shape}')  # (N,)

        topk = int(len(attn_score_per_node) * topk_ratio) # 取注意力分数前百分之十的超点索引
        topk_indices = torch.topk(attn_score_per_node, topk).indices.cpu().numpy()  # shape: (topk,)
        print(f'[Debug] topk_indices shape: {topk_indices.shape}')  # (topk,)

        # 提取超点及其 patch
        selected_superpoints = ref_points_c_y[topk_indices]  # (topk, 3)
        selected_patches = ref_node_knn_points[topk_indices].reshape(-1, 3)  # (topk*K, 3)
        print(f'[Debug] selected_superpoints shape: {selected_superpoints.shape}')
        print(f'[Debug] selected_patches shape: {selected_patches.shape}')

        # 给不同来源的点赋颜色
        superpoint_color = np.tile(np.array([[1.0, 0.0, 0.0]]), (selected_superpoints.shape[0], 1))  # 红色
        patch_color = np.tile(np.array([[0.0, 1.0, 0.0]]), (selected_patches.shape[0], 1))  # 绿色

        # 拼接点和颜色
        all_points = np.concatenate([selected_superpoints, selected_patches], axis=0)
        all_colors = np.concatenate([superpoint_color, patch_color], axis=0)

        # 创建 point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)

        # 保存 ply 文件
        save_path = os.path.join(save_dir, f'attn_ref_points_layer{lid+1}.ply')
        o3d.io.write_point_cloud(save_path, pcd)
        print(f'[Save] Saved layer {lid+1} high-attention patch to: {save_path}')


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
        if iteration < 12:
            save_high_attention_superpoints_and_patches(
                output_dict, 
                layer_ids=[5, 6], 
                iteration=iteration,
                topk_ratio=0.3
            )

            
            

        


        #attention_scores = release_cuda(output_dict['attention_scores'])

        #print(f'[Debug] attention_scores shape: {attention_scores.shape}')


def main():
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()


if __name__ == '__main__':
    main()

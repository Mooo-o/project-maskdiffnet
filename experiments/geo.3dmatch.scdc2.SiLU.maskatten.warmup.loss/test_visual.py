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

from open3d import geometry, utility, io
import os
from vedo import Points, Lines, Plotter, Text2D
import open3d as o3d

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
        )
        
        # ==================================================================
        # 点对匹配代码
        # ==================================================================
        # 1. 准备数据 (Tensor -> Numpy)
        ref_corr_points = release_cuda(output_dict['ref_corr_points'])
        src_corr_points = release_cuda(output_dict['src_corr_points'])
        transform_gt = release_cuda(data_dict['transform'])
        
        # 注意：你原代码里用的是 data_dict['ref_corr_points'] 作为点云保存
        # 但通常可视化背景点云应该用 ref_points (密集) 或 ref_points_c (稀疏)
        # ref_corr_points 只是匹配点，数量很少
        # 这里我假设你想保存的是密集点云作为背景
        ref_points_cloud = release_cuda(output_dict['ref_points']) 
        src_points_cloud = release_cuda(output_dict['src_points'])
        # ==================================================================
        
        
        # ==================================================================
        # [修正版] 在 eval_step 中完整的可视化代码
        # ==================================================================

        # 1. 安全获取指标 (转为 Python float)
        raw_ir = result_dict.get('IR', 0.0)
        inlier_ratio = raw_ir.item() if hasattr(raw_ir, 'item') else float(raw_ir)
        
        raw_ov = data_dict.get('overlap', 0.0)
        overlap = raw_ov.item() if hasattr(raw_ov, 'item') else float(raw_ov)

        self.logger.info(f"Check: {data_dict['scene_name']} IR={inlier_ratio:.3f}, OV={overlap:.3f}")

        # 2. 筛选条件
        INLIER_THRESHOLD = 0.9
        OVERLAP_MIN = 0.2
        OVERLAP_MAX = 0.4
       
        
        scene_name = data_dict['scene_name']
        ref_id = data_dict['ref_frame']
        src_id = data_dict['src_frame']
        base_name = f"{scene_name}_{ref_id}_{src_id}_IR{inlier_ratio:.2f}_OV{overlap:.2f}"
        # 指定你想要的那一对
        TARGET_SCENE = "7-scenes-redkitchen"
        TARGET_REF = 3
        TARGET_SRC = 46
        
        
        '''# ==================================================================
        # 点对匹配代码
        # ==================================================================
        if inlier_ratio > INLIER_THRESHOLD and overlap < 0.25:
                
            # --------------------------------------------------
            # 这里放原来的导出代码 PLY + NPZ
            # --------------------------------------------------
            export_dir = "vedo_visual"
            os.makedirs(export_dir, exist_ok=True)
            
            scene_name = data_dict['scene_name']
            ref_id = data_dict['ref_frame']
            src_id = data_dict['src_frame']
            base_name = f"{scene_name}_{ref_id}_{src_id}_IR{inlier_ratio:.2f}_OV{overlap:.2f}"

            # 4. 导出点云 (Background)
            # Ref
            pcd_ref = o3d.geometry.PointCloud()
            pcd_ref.points = o3d.utility.Vector3dVector(ref_points_cloud)
            # 可以在这里染成灰色，防止之后 vedo 加载是黑的
            pcd_ref.paint_uniform_color([1.0, 0.95, 0.65]) 
            o3d.io.write_point_cloud(os.path.join(export_dir, base_name + "_ref.ply"), pcd_ref)

            # Src
            pcd_src = o3d.geometry.PointCloud()
            pcd_src.points = o3d.utility.Vector3dVector(src_points_cloud)
            pcd_src.paint_uniform_color([0.53, 0.81, 0.92])
            o3d.io.write_point_cloud(os.path.join(export_dir, base_name + "_src.ply"), pcd_src)

            # 5. 计算 Inlier Mask (用于连线颜色)
            DIST_THRESH = 0.1
            R_gt = transform_gt[:3, :3]
            t_gt = transform_gt[:3, 3]
            
            # src 匹配点变换后 与 ref 匹配点 的距离
            src_corr_trans = (R_gt @ src_corr_points.T).T + t_gt
            errors = np.linalg.norm(src_corr_trans - ref_corr_points, axis=1)
            mask_inlier = errors < DIST_THRESH

            # 6. 保存匹配数据 (用于画线)
            np.savez(
                os.path.join(export_dir, base_name + "_corr.npz"),
                ref_corr_points=ref_corr_points,
                src_corr_points=src_corr_points,
                mask_inlier=mask_inlier,
                overlap=overlap,
                inlier_ratio=inlier_ratio,
                transform=transform_gt # 把 GT 姿态也存一下，万一要用
            )
            
            self.logger.info(f"Exported good match: {base_name}")
        # ==================================================================
        '''
        
        
        #if inlier_ratio > INLIER_THRESHOLD and OVERLAP_MIN < overlap < OVERLAP_MAX:
        if scene_name == TARGET_SCENE and ref_id == TARGET_REF and src_id == TARGET_SRC:
            import open3d as o3d
            import os
            
            vis_dir = "atten_visual_v2"
            os.makedirs(vis_dir, exist_ok=True)

            '''scene_name = data_dict['scene_name']
            ref_id = data_dict['ref_frame']
            src_id = data_dict['src_frame']
            '''
            base_name = f"{scene_name}_{ref_id}_{src_id}"

            # 准备数据
            ref_points = release_cuda(output_dict['ref_points']) # (N, 3)
            src_points = release_cuda(output_dict['src_points']) # (M, 3)
            transform_gt = release_cuda(data_dict['transform'])  # (4, 4)
            
            # =========================================================
            # A) 带有重叠区域标注的密集点云 (Overlap Highlight)
            # =========================================================
            DIST_THRESH = 0.05 
            
            # 1. 计算 Source 重叠掩码
            R_gt = transform_gt[:3, :3]
            t_gt = transform_gt[:3, 3]
            src_points_trans = (R_gt @ src_points.T).T + t_gt
            
            pcd_ref_tmp = o3d.geometry.PointCloud()
            pcd_ref_tmp.points = o3d.utility.Vector3dVector(ref_points)
            pcd_ref_tree = o3d.geometry.KDTreeFlann(pcd_ref_tmp)
            
            src_overlap_mask = np.zeros(src_points.shape[0], dtype=bool)
            for i, pt in enumerate(src_points_trans):
                [_, idx, dist_sq] = pcd_ref_tree.search_knn_vector_3d(pt, 1)
                if dist_sq[0] < DIST_THRESH**2:
                    src_overlap_mask[i] = True
            
            # 2. 计算 Reference 重叠掩码
            pcd_src_trans_tmp = o3d.geometry.PointCloud()
            pcd_src_trans_tmp.points = o3d.utility.Vector3dVector(src_points_trans)
            pcd_src_tree = o3d.geometry.KDTreeFlann(pcd_src_trans_tmp)
            
            ref_overlap_mask = np.zeros(ref_points.shape[0], dtype=bool)
            for i, pt in enumerate(ref_points):
                [_, idx, dist_sq] = pcd_src_tree.search_knn_vector_3d(pt, 1)
                if dist_sq[0] < DIST_THRESH**2:
                    ref_overlap_mask[i] = True
            
            # 3. 保存 Base (Color Coded)
            COLOR_NON_OVERLAP = [0.8, 0.8, 0.8] # 灰
            COLOR_OVERLAP     = [1.0, 1.0, 0.7] # 浅黄
            
            # Ref Base
            colors_ref = np.tile(COLOR_NON_OVERLAP, (ref_points.shape[0], 1))
            colors_ref[ref_overlap_mask] = COLOR_OVERLAP
            pcd_ref = o3d.geometry.PointCloud()
            pcd_ref.points = o3d.utility.Vector3dVector(ref_points)
            pcd_ref.colors = o3d.utility.Vector3dVector(colors_ref)
            o3d.io.write_point_cloud(osp.join(vis_dir, base_name + "_ref_base.ply"), pcd_ref)

            # Src Base
            colors_src = np.tile(COLOR_NON_OVERLAP, (src_points.shape[0], 1))
            colors_src[src_overlap_mask] = COLOR_OVERLAP
            pcd_src = o3d.geometry.PointCloud()
            pcd_src.points = o3d.utility.Vector3dVector(src_points)
            pcd_src.colors = o3d.utility.Vector3dVector(colors_src)
            o3d.io.write_point_cloud(osp.join(vis_dir, base_name + "_src_base.ply"), pcd_src)

            # =========================================================
            # B) 注意力高亮点 (灰 -> 深蓝)
            # =========================================================
            ref_corr_points = release_cuda(output_dict['ref_corr_points'])
            src_corr_points = release_cuda(output_dict['src_corr_points'])
            scores = release_cuda(output_dict['corr_scores'])

            # 分数归一化
            if scores.size > 0:
                scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
            else:
                scores_norm = scores # 空数组处理
                
            # ------------------ [修改开始] ------------------
            # 定义颜色 (RGB 0-1)
            # 灰色 (Low Score) - 稍微带一点冷色调的灰，更干净
            COLOR_LOW = np.array([0.85, 0.85, 0.90]) 
            
            # 深宝蓝 (High Score) - Deep Royal Blue
            COLOR_HIGH = np.array([0.05, 0.25, 0.80]) 

            atten_colors = np.zeros((len(scores_norm), 3))
            
            # 向量化插值 (比 for 循环更快)
            # 公式: Color = Low * (1-s) + High * s
            # 利用 numpy 广播机制直接算
            if len(scores_norm) > 0:
                s = scores_norm[:, np.newaxis] # (N, 1)
                atten_colors = COLOR_LOW * (1 - s) + COLOR_HIGH * s
            # ------------------ [修改结束] ------------------
            
            #atten_colors = np.zeros((len(scores_norm), 3))
            #for i, s in enumerate(scores_norm):
            #    atten_colors[i] = [0.8 * (1-s), 0.8 * (1-s), 0.8 + 0.2 * s]

            pcd_ref_atten = o3d.geometry.PointCloud()
            pcd_ref_atten.points = o3d.utility.Vector3dVector(ref_corr_points)
            pcd_ref_atten.colors = o3d.utility.Vector3dVector(atten_colors)
            o3d.io.write_point_cloud(osp.join(vis_dir, base_name + "_ref_atten.ply"), pcd_ref_atten)

            pcd_src_atten = o3d.geometry.PointCloud()
            pcd_src_atten.points = o3d.utility.Vector3dVector(src_corr_points)
            pcd_src_atten.colors = o3d.utility.Vector3dVector(atten_colors)
            o3d.io.write_point_cloud(osp.join(vis_dir, base_name + "_src_atten.ply"), pcd_src_atten)

            # 元信息
            np.savez(osp.join(vis_dir, base_name + "_meta.npz"), inlier_ratio=inlier_ratio, overlap=overlap)

            self.logger.info(f"[AttenVis] Saved {base_name} to {vis_dir}")
        

        

        

        '''
            # 1) 灰色密集点云
            ref_points = release_cuda(output_dict['ref_points'])
            src_points = release_cuda(output_dict['src_points'])

            pcd_ref = o3d.geometry.PointCloud()
            pcd_ref.points = o3d.utility.Vector3dVector(ref_points)
            pcd_ref.paint_uniform_color([0.8, 0.8, 0.8])
            o3d.io.write_point_cloud(
                osp.join(vis_dir, base_name + "_ref_base.ply"), pcd_ref
            )

            pcd_src = o3d.geometry.PointCloud()
            pcd_src.points = o3d.utility.Vector3dVector(src_points)
            pcd_src.paint_uniform_color([0.8, 0.8, 0.8])
            o3d.io.write_point_cloud(
                osp.join(vis_dir, base_name + "_src_base.ply"), pcd_src
            )
            '''

            
        
            


def main():
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()


if __name__ == '__main__':
    main()
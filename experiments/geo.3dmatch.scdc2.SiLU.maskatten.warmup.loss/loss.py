import torch
import torch.nn as nn
import torch.nn.functional as F


from geotransformer.modules.loss import WeightedCircleLoss
from geotransformer.modules.ops.transformation import apply_transform
from geotransformer.modules.registration.metrics import isotropic_transform_error
from geotransformer.modules.ops.pairwise_distance import pairwise_distance

def compute_covariance(points: torch.Tensor) -> torch.Tensor:
    """
    points: (N, 3)
    return: (3, 3) covariance matrix
    """
    if points.shape[0] < 2 or torch.isnan(points).any() or torch.isinf(points).any():
        return torch.zeros(3, 3, device=points.device)

    centroid = points.mean(dim=0, keepdim=True)
    centered = points - centroid
    cov = centered.T @ centered / (points.shape[0] - 1)
    if torch.isnan(cov).any() or torch.isinf(cov).any():
        return torch.zeros(3, 3, device=points.device)
    return cov


def principal_direction_angle_loss(cov_p: torch.Tensor, cov_q: torch.Tensor) -> torch.Tensor:
    """
    Compute angle-based loss between principal directions of two covariance matrices.
    Return a scalar in [0, 1]
    """
    try:
        eigval_p, eigvec_p = torch.linalg.eigh(cov_p)
        eigval_q, eigvec_q = torch.linalg.eigh(cov_q)
    except RuntimeError as e:
        print("[WARN] eig decomposition failed:", e)
        return torch.tensor(0.0, device=cov_p.device)

    v1_p = eigvec_p[:, -1]  # principal eigenvector
    v1_q = eigvec_q[:, -1]

    cos_sim = torch.abs(F.cosine_similarity(v1_p, v1_q, dim=0))
    if torch.isnan(cos_sim):
        return torch.tensor(0.0, device=cov_p.device)
    return 1 - cos_sim


def geometric_direction_consistency_loss(patch_p, patch_q, mask=None, min_valid_points=4):
    assert patch_p.shape == patch_q.shape
    N = patch_p.shape[0]
    device = patch_p.device

    if mask is None:
        mask = torch.ones((N, patch_p.shape[1]), dtype=torch.bool, device=device)

    total_loss = 0.0
    count = 0

    for i in range(N):
        valid_mask = mask[i]
        if valid_mask.sum() < min_valid_points:
            continue
        pp = patch_p[i][valid_mask]
        pq = patch_q[i][valid_mask]

        if torch.isnan(pp).any() or torch.isnan(pq).any():
            continue

        cov_p = compute_covariance(pp)
        cov_q = compute_covariance(pq)
        loss_i = principal_direction_angle_loss(cov_p, cov_q)

        if not torch.isnan(loss_i):
            total_loss += loss_i
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=device)
    return total_loss / count



'''def geometric_consistency_loss_flat(patch_p, patch_q, mask=None):
    """
    patch_p: (N, K, 3)
    patch_q: (N, K, 3)
    mask: (N, K) bool mask indicating valid points (optional)
    """
    assert patch_p.shape == patch_q.shape
    N, K, _ = patch_p.shape
    device = patch_p.device

    if mask is None:
        mask = torch.ones(N, K, dtype=torch.bool, device=device)

    min_valid_points = 4
    skipped = 0
    count = 0
    loss = 0.0
    
    for i in range(N):
        valid_mask = mask[i]
        #valid_num = valid_mask.sum().item()
        if valid_mask.sum() < min_valid_points:
            #print(f"[DEBUG] patch {i} skipped, valid points = {valid_num}")
            skipped += 1
            continue
        pp = patch_p[i][valid_mask]
        pq = patch_q[i][valid_mask]
        cov_p = compute_covariance(pp)
        cov_q = compute_covariance(pq)
        diff = torch.norm(cov_p - cov_q, p='fro') ** 2
        if diff < 1e-6:
            print(f"[DEBUG] patch {i} cov diff ~0")
        else:
            print(f"[DEBUG] patch {i} cov diff = {diff.item():.6f}")

        mean_dist = torch.norm(pp.mean(dim=0) - pq.mean(dim=0))
        print(f"[DEBUG] patch {i} mean center dist: {mean_dist:.4f}")

        loss += torch.norm(cov_p - cov_q, p='fro') ** 2
        count += 1
    
    print(f"[DEBUG] geo_loss used patches: {count}, skipped: {skipped}")
    return loss / count if count > 0 else torch.tensor(0.0, device=patch_p.device)'''

class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,
        )
        self.positive_overlap = cfg.coarse_loss.positive_overlap
        self.weight_geo_loss = getattr(cfg.coarse_loss, 'weight_geo_loss', 0.1)
        
        self.geo_loss_not_zero_count = 0

    def forward(self, output_dict):
        ref_feats = output_dict['ref_feats_c']
        src_feats = output_dict['src_feats_c']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))

        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        if 'ref_node_corr_knn_points' in output_dict and 'src_node_corr_knn_points' in output_dict:
            ref_patches = output_dict['ref_node_corr_knn_points']  # (N, K, 3)
            src_patches = output_dict['src_node_corr_knn_points']  # (N, K, 3)
        
            if 'ref_node_corr_knn_masks' in output_dict and 'src_node_corr_knn_masks' in output_dict:
                ref_mask = output_dict['ref_node_corr_knn_masks']  # (N, K)
                src_mask = output_dict['src_node_corr_knn_masks']
                mask = ref_mask & src_mask
            else:
                mask = None
        
            geo_loss = geometric_direction_consistency_loss(ref_patches, src_patches, mask)
            print(f"[DEBUG] geometric direction loss: {geo_loss.item():.6f}")
            loss += self.weight_geo_loss * geo_loss
            print(f"[DEBUG] CoarseMatchingLoss: {loss.item():.6f}")

        
        return loss

class FineMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(FineMatchingLoss, self).__init__()
        self.positive_radius = cfg.fine_loss.positive_radius

    def forward(self, output_dict, data_dict):
        ref_node_corr_knn_points = output_dict['ref_node_corr_knn_points']
        src_node_corr_knn_points = output_dict['src_node_corr_knn_points']
        ref_node_corr_knn_masks = output_dict['ref_node_corr_knn_masks']
        src_node_corr_knn_masks = output_dict['src_node_corr_knn_masks']
        matching_scores = output_dict['matching_scores']
        transform = data_dict['transform']

        # 将 src 点云变换到 ref 坐标系
        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)
        dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  # (B, N, M)

        gt_masks = torch.logical_and(
            ref_node_corr_knn_masks.unsqueeze(2).expand(-1, -1, src_node_corr_knn_masks.shape[1]),
            src_node_corr_knn_masks.unsqueeze(1).expand(-1, ref_node_corr_knn_masks.shape[1], -1)
        )

        gt_corr_map = torch.lt(dists, self.positive_radius ** 2)
        gt_corr_map = torch.logical_and(gt_corr_map, gt_masks)  # (B, N, M)

        loss = self.info_nce_matching_loss(matching_scores, gt_corr_map)
        print(f"[DEBUG] FineMatchingLoss: {loss.item():.6f}")
        return loss

    def info_nce_matching_loss(self, matching_scores, gt_corr_map):
        """
        对每个 batch 中的正匹配对 (i, j)，让 matching_scores[i, :] 以 j 为标签做 cross entropy。
        """
        B, N, M = gt_corr_map.shape
        loss = 0.0
        valid_batch_count = 0
    
        for b in range(B):
            pos_mask = gt_corr_map[b]  # shape: (N, M)
    
            if pos_mask.sum() == 0:
                continue
    
            pos_idx = torch.nonzero(pos_mask, as_tuple=False)  # (P, 2)
            ref_idx = pos_idx[:, 0]
            src_idx = pos_idx[:, 1]
    
            logits = matching_scores[b, :-1, :-1]  # shape: (N, M)
    
            # 筛掉超出范围的标签对（以防 src_idx >= M）
            valid_mask = (ref_idx < logits.shape[0]) & (src_idx < logits.shape[1])
            if valid_mask.sum() == 0:
                continue
    
            ref_idx = ref_idx[valid_mask]
            src_idx = src_idx[valid_mask]
    
            selected_logits = logits[ref_idx]   # shape: (P, M)
            targets = src_idx                   # shape: (P,)
    
            # 安全断言调试（训练正常时可注释掉）
            assert targets.max() < selected_logits.shape[1], \
                f"[ERROR] Target index {targets.max().item()} out of range for logits of shape {selected_logits.shape}"
    
            # Cross-entropy (InfoNCE)
            loss_b = F.cross_entropy(selected_logits, targets, reduction='mean')
            loss += loss_b
            valid_batch_count += 1
    
        if valid_batch_count > 0:
            loss /= valid_batch_count
        else:
            loss = torch.tensor(0.0, device=matching_scores.device, requires_grad=True)
    
        return loss


'''class FineMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(FineMatchingLoss, self).__init__()
        self.positive_radius = cfg.fine_loss.positive_radius

    def forward(self, output_dict, data_dict):
        ref_node_corr_knn_points = output_dict['ref_node_corr_knn_points']
        src_node_corr_knn_points = output_dict['src_node_corr_knn_points']
        ref_node_corr_knn_masks = output_dict['ref_node_corr_knn_masks']
        src_node_corr_knn_masks = output_dict['src_node_corr_knn_masks']
        matching_scores = output_dict['matching_scores']
        transform = data_dict['transform']
        
        #print(f"[DEBUG] matching_scores: {matching_scores.shape}")
        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)
        dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  # (B, N, M)
        # gt_masks = torch.logical_and(ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1))
        # 改动
        gt_masks = torch.logical_and(
            ref_node_corr_knn_masks.unsqueeze(2).expand(-1, -1, src_node_corr_knn_masks.shape[1]),
            src_node_corr_knn_masks.unsqueeze(1).expand(-1, ref_node_corr_knn_masks.shape[1], -1)
        )

        
        gt_corr_map = torch.lt(dists, self.positive_radius ** 2)
        gt_corr_map = torch.logical_and(gt_corr_map, gt_masks)

        loss = self.info_nce_matching_loss(matching_scores, gt_corr_map)
        print(f"[DEBUG] FineMatchingLoss: {loss.item():.6f}")
        return loss

    def info_nce_matching_loss(self, matching_scores, gt_corr_map):
        """
        对每个 batch 中的正匹配对 (i, j)，让 matching_scores[i, :] 以 j 为标签做 cross entropy。
        """
        B, N, M = gt_corr_map.shape
        loss = 0.0
        valid_batch_count = 0

        for b in range(B):
            pos_mask = gt_corr_map[b]  # shape: (N, M)

            if pos_mask.sum() == 0:
                continue

            pos_idx = torch.nonzero(pos_mask, as_tuple=False)  # (P, 2)
            ref_idx = pos_idx[:, 0]
            src_idx = pos_idx[:, 1]

            logits = matching_scores[b, :-1, :-1]  # shape: (N, M)

            # 每个 ref_idx 对应 logits 的一行
            selected_logits = logits[ref_idx]           # (P, M)
            targets = src_idx                          # (P,)

            # 交叉熵损失（InfoNCE）
            loss_b = F.cross_entropy(selected_logits, targets, reduction='mean')
            loss += loss_b
            valid_batch_count += 1

        if valid_batch_count > 0:
            loss /= valid_batch_count
        else:
            # fallback: 不影响梯度传播
            loss = torch.tensor(0.0, device=matching_scores.device, requires_grad=True)

        return loss '''


class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.fine_loss = FineMatchingLoss(cfg)
        self.weight_coarse_loss = cfg.loss.weight_coarse_loss
        self.weight_fine_loss = cfg.loss.weight_fine_loss

    def forward(self, output_dict, data_dict):
        coarse_loss = self.coarse_loss(output_dict)
        fine_loss = self.fine_loss(output_dict, data_dict)

        loss = self.weight_coarse_loss * coarse_loss + self.weight_fine_loss * fine_loss

        return {
            'loss': loss,
            'c_loss': coarse_loss,
            'f_loss': fine_loss,
        }


class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rmse = cfg.eval.rmse_threshold

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        ref_length_c = output_dict['ref_points_c'].shape[0]
        src_length_c = output_dict['src_points_c'].shape[0]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(ref_length_c, src_length_c).cuda()
        gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.0

        ref_node_corr_indices = output_dict['ref_node_corr_indices']
        src_node_corr_indices = output_dict['src_node_corr_indices']

        precision = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices].mean()

        return precision

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        transform = data_dict['transform']
        ref_corr_points = output_dict['ref_corr_points']
        src_corr_points = output_dict['src_corr_points']

        if ref_corr_points.numel() == 0 or src_corr_points.numel() == 0:
            return torch.tensor(0.0, device=transform.device)

        try:
            src_corr_points = apply_transform(src_corr_points, transform)
        except Exception as e:
            print("[DEBUG] apply_transform failed:", e)
            return torch.tensor(0.0, device=transform.device)

        try:
            corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)
        except Exception as e:
            print("[DEBUG] corr_distances error:", e)
            return torch.tensor(0.0, device=transform.device)

        if corr_distances.numel() == 0:
            return torch.tensor(0.0, device=transform.device)

        if torch.isnan(corr_distances).any():
            print("[DEBUG] corr_distances contain NaNs")
            return torch.tensor(0.0, device=transform.device)

        precision = torch.lt(corr_distances, self.acceptance_radius).float().mean()
        if torch.isnan(precision):
            print("[DEBUG] IR is NaN")
            print(f"ref_corr_points: {ref_corr_points.shape}")
            print(f"src_corr_points: {src_corr_points.shape}")
            print(f"corr_distances: {corr_distances}")
            return torch.tensor(0.0, device=transform.device)

        return precision

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        transform = data_dict['transform']
        est_transform = output_dict['estimated_transform']
        src_points = output_dict['src_points']

        rre, rte = isotropic_transform_error(transform, est_transform)

        realignment_transform = torch.matmul(torch.inverse(transform), est_transform)
        realigned_src_points_f = apply_transform(src_points, realignment_transform)
        rmse = torch.linalg.norm(realigned_src_points_f - src_points, dim=1).mean()
        recall = torch.lt(rmse, self.acceptance_rmse).float()

        return rre, rte, rmse, recall

    def forward(self, output_dict, data_dict):
        c_precision = self.evaluate_coarse(output_dict)
        f_precision = self.evaluate_fine(output_dict, data_dict)
        rre, rte, rmse, recall = self.evaluate_registration(output_dict, data_dict)

        return {
            'PIR': c_precision,
            'IR': f_precision,
            'RRE': rre,
            'RTE': rte,
            'RMSE': rmse,
            'RR': recall,
        }

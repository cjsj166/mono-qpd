import torch
import numpy as np


# ------------------------------------------------------------
# Dummy spearman_correlation 함수 (실제 구현이 없다면 아래처럼 대체)
def spearman_correlation(Y, Target):
    # 여기서는 단순히 배치별 임의 값을 반환 (실제 구현 시에는 순위 기반 상관계수를 계산)
    B = Y.shape[0]
    return torch.rand(B)

# ------------------------------------------------------------
# torch 기반 affine invariant 함수 (배치 처리)
def affine_invariant_1_batch(
        Y: torch.Tensor,
        Target: torch.Tensor,
        confidence_map: torch.Tensor = None,
        irls_iters: int = 5,
        eps: float = 1e-3):
    """
    입력 Y, Target, confidence_map은 shape (B, C, H, W)를 가지며,
    각 배치별로 affine invariant 1 metric과 affine 파라미터 b (scale, bias)를 계산합니다.
    반환: ai1_arr: (B,), b_arr: (B, 2)
    """
    B = Y.shape[0]
    ai1_list = []
    b_list = []
    for i in range(B):
        y = Y[i].reshape(-1)  # [N,]
        t = Target[i].reshape(-1)  # [N,]
        if confidence_map is None:
            conf = torch.ones_like(t, dtype=torch.float)
        else:
            conf = confidence_map[i].reshape(-1)
        
        # 초기 IRLS weight
        w = torch.ones_like(y, dtype=torch.float)
        ones = torch.ones_like(y, dtype=torch.float)
        for _ in range(irls_iters):
            w_sqrt = torch.sqrt(w * conf)  # [N,]
            WX = w_sqrt.unsqueeze(1) * torch.stack([y, ones], dim=1)  # [N, 2]
            Wt = w_sqrt * t  # [N,]
            sol = torch.linalg.lstsq(WX, Wt)
            b = sol.solution  # (2,)
            affine_y = y * b[0] + b[1]
            residual = torch.abs(affine_y - t)
            w = 1 / torch.maximum(torch.tensor(eps, dtype=torch.float, device=w.device), residual)
        ai1 = torch.sum(conf * residual) / torch.sum(conf)
        ai1_list.append(ai1)
        b_list.append(b)
    
    ai1_arr = torch.stack(ai1_list, dim=0)  # (B,)
    b_arr = torch.stack(b_list, dim=0)  # (B, 2)
    return ai1_arr, b_arr


def affine_invariant_2_batch(
        Y: torch.Tensor,
        Target: torch.Tensor,
        confidence_map: torch.Tensor = None,
        eps: float = 1e-3):
    """
    입력 Y, Target, confidence_map은 shape (B, C, H, W)를 가지며,
    각 배치별로 affine invariant 2 metric과 affine 파라미터 b (scale, bias)를 계산합니다.
    반환: ai2_arr: (B,), b_arr: (B, 2)
    """
    B = Y.shape[0]
    ai2_list = []
    b_list = []
    for i in range(B):
        y = Y[i].reshape(-1)  # [N,]
        t = Target[i].reshape(-1)  # [N,]
        if confidence_map is None:
            conf = torch.ones_like(t, dtype=torch.float)
        else:
            conf = confidence_map[i].reshape(-1)
        
        ones = torch.ones_like(y, dtype=torch.float)
        X = conf.unsqueeze(1) * torch.stack([y, ones], dim=1)  # [N, 2]
        t_conf = conf * t  # [N,]
        sol = torch.linalg.lstsq(X, t_conf)
        b = sol.solution  # (2,)
        affine_y = y * b[0] + b[1]
        max_val = torch.tensor(np.finfo(np.float32).max, dtype=torch.float, device=affine_y.device)
        residual_sq = torch.minimum((affine_y - t_conf) ** 2, max_val)
        ai2 = torch.sqrt(torch.sum(conf * residual_sq) / torch.sum(conf))
        ai2_list.append(ai2)
        b_list.append(b)
    
    ai2_arr = torch.stack(ai2_list, dim=0)  # (B,)
    b_arr = torch.stack(b_list, dim=0)  # (B, 2)
    return ai2_arr, b_arr

# ------------------------------------------------------------
# torch 기반 compute_scale 함수
def compute_scale(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """
    각 배치별로 최적의 스케일(alpha)을 계산합니다.
    prediction, target, mask: shape (B, C, H, W) torch.Tensor.
    alpha = sum(mask * prediction * target) / sum(mask * prediction^2)
    반환: alpha, shape (B,)
    """
    numerator = torch.sum(mask * prediction * target, dim=(1,2,3))
    denominator = torch.sum(mask * prediction * prediction, dim=(1,2,3))
    alpha = torch.zeros_like(numerator)
    valid = denominator != 0
    alpha[valid] = numerator[valid] / denominator[valid]
    return alpha
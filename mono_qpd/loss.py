import torch
import torch.nn as nn
import numpy as np

class ScaleInvariantLoss(nn.Module):
    """
    배치(B) 크기와 임의의 공간적 차원(...)을 가진 pred, target을 입력받아,
    각 위치별(pixel별)로 scale-invariant loss를 계산한 결과를 반환합니다.
    
    - pred, target: shape (B, ...)
    - 반환값 i_loss: shape (B, ...), i_loss[b].sum() 하면 
      샘플 b의 scale-invariant loss 스칼라와 동일합니다.
    """
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: (B, ...)
        target: (B, ...)
        """
        B = pred.shape[0]
        
        # 배치마다 나머지 차원을 하나로 펼쳐서 계산
        pred_flat = pred.view(B, -1)
        target_flat = target.view(B, -1)

        # log 스케일로 변환 (log(0) 방지를 위해 epsilon 추가)
        log_pred = torch.log(pred_flat + self.epsilon)
        log_target = torch.log(target_flat + self.epsilon)

        # diff[b, i] = log_pred[b, i] - log_target[b, i]
        diff = log_pred - log_target
        
        # 한 샘플당 픽셀 수
        n = diff.shape[1]
        
        # 각 샘플별 diff 합
        diff_sum = diff.sum(dim=1)  # shape [B]
        
        # 두 번째 항 (1/n^2)*(sum_i diff_i)^2 을 샘플별로 구함
        second_term = (diff_sum ** 2) / (n ** 2)  # shape [B]
        
        # 첫 번째 항 per-pixel: diff^2 / n
        # 두 번째 항은 모든 픽셀에 동일하게 분배: second_term / n
        # i_loss_flat[b, i] = (diff[b, i]^2 / n) - (second_term[b] / n)
        i_loss_flat = (diff**2) / n - (second_term / n).unsqueeze(dim=1)
        
        # 원래 (B, ...) 모양으로 복원
        i_loss = i_loss_flat.view_as(pred)
        return i_loss
    

def compute_scale(prediction, target, mask):
    """
    각 배치별로 최적의 스케일(alpha)를 계산합니다.
    
    prediction: (B, H, W) 텐서 (또는 기타 공간 차원)
    target: (B, H, W) 텐서
    mask: (B, H, W) 텐서 (0 또는 1로 valid 영역 지정)
    
    alpha는 각 배치에 대해 아래 식을 만족합니다:
      alpha = sum(mask * prediction * target) / sum(mask * prediction^2)
    """
    # 배치별로 (H, W) 차원에서 합산합니다.
    numerator = torch.sum(mask * prediction * target, axis=(1, 2, 3))
    denominator = torch.sum(mask * prediction * prediction, axis=(1, 2, 3))
    
    # denominator가 0인 경우를 처리하기 위해 기본값은 0으로 설정합니다.
    alpha = torch.zeros_like(numerator)
    valid = (denominator != 0)
    alpha[valid] = numerator[valid] / denominator[valid]
    return alpha

class LeastSquareScaleInvariantLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, prediction, target, mask):
        """
        (d - (d' * alpha))^2 의 loss를 계산합니다.
        
        prediction: (B, C, H, W) 예측 텐서
        target: (B, C, H, W) 목표 텐서
        mask: (B, C, H, W) 마스크 텐서 (0 또는 1)
        
        배치 내 각 이미지에 대해 valid 픽셀에 대한 MSE 평균을 계산한 후,
        전체 배치 평균을 반환합니다.
        """
        # 각 배치별 최적의 스케일(alpha) 계산 (shape: (B,))
        alpha = compute_scale(prediction, target, mask)
        # alpha를 이미지 전체에 적용하기 위해 차원 확장: (B, 1, 1)
        alpha_expanded = alpha.view(-1, 1, 1, 1)
        
        # 스케일 조정된 예측: d' * alpha
        prediction_scaled = prediction * alpha_expanded
        
        # 각 픽셀별 오차 제곱: (d - (d' * alpha))^2
        pixel_loss = (target - prediction_scaled) ** 2
        
        # 마스크된 영역에 대해서만 loss를 계산
        masked_loss = mask * pixel_loss
        
        # 이미지별 평균 loss = sum(masked_loss) / sum(mask)
        loss_per_image = torch.zeros(prediction.shape[0], device=prediction.device)
        sum_mask = torch.sum(mask, dim=(1,2,3))
        valid = sum_mask != 0
        loss_per_image[valid] = torch.sum(masked_loss, dim=(1,2,3))[valid] / sum_mask[valid]
        
        # 전체 배치의 평균 loss 반환
        return loss_per_image.mean()
    

def test_scale_invariant_loss():
    B, C, H, W = 2, 1, 224, 224
    pred = torch.rand(B, C, H, W)
    target = torch.rand(B, C, H, W)
    loss_fn = ScaleInvariantLoss()
    loss = loss_fn(pred, target)
    assert loss.shape == pred.shape, "Loss shape mismatch"
    print("ScaleInvariantLoss test passed")

def test_least_square_scale_invariant_loss():
    B, C, H, W = 2, 1, 224, 224
    pred = torch.rand(B, C, H, W)
    target = torch.rand(B, C, H, W)
    mask = torch.ones(B, C, H, W)
    loss_fn = LeastSquareScaleInvariantLoss()
    loss = loss_fn(pred, target, mask)
    assert loss.dim() == 0, "Loss should be a scalar"
    print("LeastSquareScaleInvariantLoss test passed")

if __name__ == "__main__":
    test_scale_invariant_loss()
    test_least_square_scale_invariant_loss()

import torch
import os.path as osp
from datetime import datetime
import numpy as np
# from metrics.affine_invariant_metrics import *
from affine_invariant_metrics import * # for testing

# ------------------------------------------------------------
# Eval 클래스 (모든 연산을 torch.Tensor 기반, 입력 shape: (B, C, H, W))
class Eval():
    def __init__(self, save_path='', enabled_metrics=None):
        if enabled_metrics is None:
            enabled_metrics = []
        self.enabled_metrics = enabled_metrics.copy()
        self.metrics_data = {metric: [] for metric in self.enabled_metrics}
        
        # 추가 지표 키 설정
        if 'ai1' in self.enabled_metrics:
            self.enabled_metrics.extend(['ai1-scale', 'ai1-bias'])
            self.metrics_data['ai1-scale'] = []
            self.metrics_data['ai1-bias'] = []
        if 'ai2' in self.enabled_metrics:
            self.enabled_metrics.extend(['ai2-scale', 'ai2-bias'])
            self.metrics_data['ai2-scale'] = []
            self.metrics_data['ai2-bias'] = []
        if 'si' in self.enabled_metrics:
            self.enabled_metrics.extend(['si', 'si-scale'])
            self.metrics_data['si'] = []
            self.metrics_data['si-scale'] = []
        
        # 기타 지표('sc', 'epe', 'rmse', 'epe_bad_*', 'ai2_bad_*')는 enabled_metrics에 포함되어 있다고 가정
        self.filenames = []
        self.color_range = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_path = save_path + '_eval_' + timestamp + '.txt'
        
    def add_colorrange(self, vmin, vmax):
        self.color_range.append((vmin, vmax))
        
    def add_filename(self, filename):
        self.filenames.append(filename)
        
    def affine_invariant_1(self, Y, Target, confidence_map=None, irls_iters=5, eps=1e-3):
        if 'ai1' in self.enabled_metrics:
            ai1_tensor, b_tensor = affine_invariant_1_batch(Y, Target, confidence_map, irls_iters, eps)
            for a, b in zip(ai1_tensor, b_tensor):
                self.metrics_data.setdefault('ai1', []).append(a.item())
                self.metrics_data['ai1-scale'].append(b[0].item())
                self.metrics_data['ai1-bias'].append(b[1].item())
            return ai1_tensor, b_tensor
        return None, None
    
    def affine_invariant_2(self, Y, Target, confidence_map=None, eps=1e-3):
        if 'ai2' in self.enabled_metrics:
            ai2_tensor, b_tensor = affine_invariant_2_batch(Y, Target, confidence_map, eps)
            for a, b in zip(ai2_tensor, b_tensor):
                self.metrics_data.setdefault('ai2', []).append(a.item())
                self.metrics_data['ai2-scale'].append(b[0].item())
                self.metrics_data['ai2-bias'].append(b[1].item())
            return ai2_tensor, b_tensor
        return None, None
    
    def scale_invariant(self, Y, Target, mask=None):
        if mask is None:
            mask = torch.ones_like(Y)
        if 'si' in self.enabled_metrics:
            alpha = compute_scale(Y, Target, mask)  # (B,)
            alpha_reshaped = alpha.view(-1, 1, 1, 1)
            si_tensor = torch.mean(((Target - alpha_reshaped * Y) ** 2), dim=(1,2,3))
            for s, a in zip(si_tensor, alpha):
                self.metrics_data['si'].append(s.item())
                self.metrics_data['si-scale'].append(a.item())
            return si_tensor, alpha
        return None
    
    def spearman_correlation(self, Y, Target):
        if 'sc' in self.enabled_metrics:
            sc_tensor = spearman_correlation(Y, Target)  # (B,)
            for s in sc_tensor:
                self.metrics_data.setdefault('sc', []).append(s.item())
            return sc_tensor
        return None
    
    def bad_pixel_metric(self, Y, Target, threshold):
        diff = torch.abs(Y - Target)
        bad_pixels = (diff > threshold).float().sum(dim=(1,2,3))
        total_pixels = Y.shape[1] * Y.shape[2] * Y.shape[3]
        ratio = bad_pixels / total_pixels  # (B,)
        return ratio.mean().item()  # 평균값을 반환
    
    def end_point_error(self, Y, Target):
        if 'epe' in self.enabled_metrics:
            epe_tensor = torch.mean(torch.abs(Y - Target), dim=(1,2,3))
            for e in epe_tensor:
                self.metrics_data.setdefault('epe', []).append(e.item())
            return epe_tensor
        return None
    
    def root_mean_squared_error(self, Y, Target):
        if 'rmse' in self.enabled_metrics:
            rmse_tensor = torch.sqrt(torch.mean((Y - Target) ** 2, dim=(1,2,3)))
            for r in rmse_tensor:
                self.metrics_data.setdefault('rmse', []).append(r.item())
            return rmse_tensor
        return None
    
    def epe_bad_pixel_metrics(self, Y, Target):
        result = {}
        if any('epe_bad' in metric for metric in self.enabled_metrics):
            _ = self.end_point_error(Y, Target)  # 계산 후 기록
            thresholds = {
                'epe_bad_0_005px': 0.005,
                'epe_bad_0_01px': 0.01,
                'epe_bad_0_05px': 0.05,
                'epe_bad_0_1px': 0.1,
                'epe_bad_0_5px': 0.5,
                'epe_bad_1px': 1.0,
                'epe_bad_3px': 3.0,
                'epe_bad_5px': 5.0,
                'epe_bad_10px': 10.0,
                'epe_bad_15px': 15.0
            }
            for key, thresh in thresholds.items():
                if key in self.enabled_metrics:
                    metric_val = self.bad_pixel_metric(Y, Target, thresh)
                    self.metrics_data.setdefault(key, []).append(metric_val)
                    result[key] = metric_val
        return result
    
    def ai2_bad_pixel_metrics(self, Y, Target):
        result = {}
        if any('bad' in metric for metric in self.enabled_metrics):
            ai2_tensor, b_tensor = self.affine_invariant_2(Y, Target)
            scale = b_tensor[:,0].view(-1, 1, 1, 1)
            bias = b_tensor[:,1].view(-1, 1, 1, 1)
            Y_corr = Y * scale + bias
            thresholds = {
                'ai2_bad_0_005px': 0.005,
                'ai2_bad_0_01px': 0.01,
                'ai2_bad_0_05px': 0.05,
                'ai2_bad_0_1px': 0.1,
                'ai2_bad_0_5px': 0.5,
                'ai2_bad_1px': 1.0,
                'ai2_bad_3px': 3.0,
                'ai2_bad_5px': 5.0,
                'ai2_bad_10px': 10.0,
                'ai2_bad_15px': 15.0
            }
            for key, thresh in thresholds.items():
                if key in self.enabled_metrics:
                    metric_val = self.bad_pixel_metric(Y_corr, Target, thresh)
                    self.metrics_data.setdefault(key, []).append(metric_val)
                    result[key] = metric_val
        return result
    
    def get_latest_metrics(self):
        latest_metrics = {metric: values[-1] for metric, values in self.metrics_data.items() if values}
        return latest_metrics
    
    def get_mean_metrics(self):
        mean_metrics = {metric: np.mean(values) for metric, values in self.metrics_data.items() if values}
        return mean_metrics
        
    def save_metrics(self):
        with open(self.save_path, "w") as f:
            header = "filename " + " ".join(self.enabled_metrics) + " color-range\n"
            f.write(header)
            for i, filename in enumerate(self.filenames):
                line = f"{filename} "
                for metric in self.enabled_metrics:
                    if metric in self.metrics_data and len(self.metrics_data[metric]) > i:
                        line += f"{self.metrics_data[metric][i]:.3f} "
                if len(self.color_range) > i:
                    line += f"{self.color_range[i][0]:.3f}-{self.color_range[i][1]:.3f}\n"
                else:
                    line += "\n"
                f.write(line)
            mean_metrics = self.get_mean_metrics()
            mean_line = "mean "
            for metric in self.enabled_metrics:
                if metric in mean_metrics:
                    mean_line += f"{mean_metrics[metric]:.3f} "
            mean_line += "\n"
            f.write(mean_line)
            f.write("----end----\n")

# ------------------------------------------------------------
# 테스트 코드: 모든 함수들을 torch.Tensor (B x C x H x W)로 생성하여 호출
if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 2
    C = 1
    H, W = 4, 4

    # 임의의 입력 텐서 생성 (B, C, H, W)
    Y = torch.rand(batch_size, C, H, W, dtype=torch.float)
    Target = torch.rand(batch_size, C, H, W, dtype=torch.float)
    mask = torch.ones_like(Y, dtype=torch.float)

    print("=== Testing compute_scale ===")
    alpha = compute_scale(Y, Target, mask)
    print("Computed scale (alpha):", alpha)

    print("\n=== Testing affine_invariant_1_batch ===")
    ai1_tensor, b1_tensor = affine_invariant_1_batch(Y, Target, confidence_map=mask, irls_iters=5, eps=1e-3)
    print("Affine Invariant 1 (per batch):", ai1_tensor)
    print("Affine parameters (scale, bias) for ai1:", b1_tensor)

    print("\n=== Testing affine_invariant_2_batch ===")
    ai2_tensor, b2_tensor = affine_invariant_2_batch(Y, Target, confidence_map=mask, eps=1e-3)
    print("Affine Invariant 2 (per batch):", ai2_tensor)
    print("Affine parameters (scale, bias) for ai2:", b2_tensor)

    print("\n=== Testing Eval.scale_invariant ===")
    # scale_invariant: 입력 Y, Target, mask (B, C, H, W)
    evaluator = Eval(save_path="test", enabled_metrics=['si'])
    si_tensor, si_alpha = evaluator.scale_invariant(Y, Target, mask)
    print("Scale Invariant Error (per sample):", si_tensor)
    print("Scale factors from scale_invariant:", si_alpha)

    print("\n=== Testing Eval.spearman_correlation ===")
    # 테스트를 위해 dummy spearman_correlation가 사용됨
    evaluator.enabled_metrics.append('sc')
    evaluator.metrics_data['sc'] = []
    sc_tensor = evaluator.spearman_correlation(Y, Target)
    print("Spearman Correlation (per batch):", sc_tensor)

    print("\n=== Testing Eval.end_point_error and root_mean_squared_error ===")
    evaluator.enabled_metrics.append('epe')
    evaluator.metrics_data['epe'] = []
    evaluator.enabled_metrics.append('rmse')
    evaluator.metrics_data['rmse'] = []
    epe_tensor = evaluator.end_point_error(Y, Target)
    rmse_tensor = evaluator.root_mean_squared_error(Y, Target)
    print("End Point Error (per sample):", epe_tensor)
    print("Root Mean Squared Error (per sample):", rmse_tensor)

    print("\n=== Testing Eval.epe_bad_pixel_metrics ===")
    # 활성화된 bad-pixel metric 키 추가
    evaluator.enabled_metrics.extend(['epe_bad_0_005px'])
    evaluator.metrics_data['epe_bad_0_005px'] = []
    epe_bad = evaluator.epe_bad_pixel_metrics(Y, Target)
    print("EPE Bad Pixel Metrics:", epe_bad)

    print("\n=== Testing Eval.ai2_bad_pixel_metrics ===")
    evaluator.enabled_metrics.extend(['ai2_bad_0_005px'])
    evaluator.metrics_data['ai2_bad_0_005px'] = []
    ai2_bad = evaluator.ai2_bad_pixel_metrics(Y, Target)
    print("AI2 Bad Pixel Metrics:", ai2_bad)

    print("\n=== Testing Eval.affine_invariant_1 and affine_invariant_2 via Eval ===")
    # 활성화된 'ai1'와 'ai2' 지표를 추가 (이미 내부에서 확장됨)
    evaluator.enabled_metrics.extend(['ai1', 'ai2'])
    evaluator.metrics_data.setdefault('ai1', [])
    evaluator.metrics_data.setdefault('ai2', [])
    _ = evaluator.affine_invariant_1(Y, Target, confidence_map=mask, irls_iters=5, eps=1e-3)
    _ = evaluator.affine_invariant_2(Y, Target, confidence_map=mask, eps=1e-3)

    # 파일명 및 컬러 범위 추가 (평가 결과 저장)
    evaluator.add_filename("dummy_image.png")
    evaluator.add_colorrange(0.0, 1.0)

    latest_metrics = evaluator.get_latest_metrics()
    mean_metrics = evaluator.get_mean_metrics()
    print("\nLatest Metrics:", latest_metrics)
    print("Mean Metrics:", mean_metrics)

    evaluator.save_metrics()
    print("Metrics saved to:", evaluator.save_path)

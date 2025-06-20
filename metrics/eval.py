from metrics.affine_invariant_metrics import *
import os.path as osp
from datetime import datetime
import numpy as np

def compute_scale(prediction, target, mask):
    """
    각 배치별로 최적의 스케일(alpha)를 계산합니다.
    
    prediction: (1, H, W) 텐서 (또는 기타 공간 차원)
    target: (1, H, W) 텐서
    mask: (1, H, W) 텐서 (0 또는 1로 valid 영역 지정)
    
    alpha는 각 배치에 대해 아래 식을 만족합니다:
      alpha = sum(mask * prediction * target) / sum(mask * prediction^2)
    """
    # 배치별로 (H, W) 차원에서 합산합니다.
    numerator = np.sum(mask * prediction * target, axis=(0, 1, 2))
    denominator = np.sum(mask * prediction * prediction, axis=(0, 1, 2))
    
    # denominator가 0인 경우를 처리하기 위해 기본값은 0으로 설정합니다.
    alpha = np.zeros_like(numerator)
    valid = (denominator != 0)
    alpha[valid] = numerator[valid] / denominator[valid]
    return alpha

class Eval():
    def __init__(self, save_path='', enabled_metrics=None):
        if enabled_metrics is None:
            enabled_metrics = []

        self.enabled_metrics = enabled_metrics
        
        self.metrics_data = {metric: [] for metric in enabled_metrics}

        # Add 'ai1-scale' and 'ai1-bias' if 'ai1' is in enabled_metrics
        if 'ai1' in enabled_metrics:
            self.enabled_metrics.append('ai1-scale')
            self.enabled_metrics.append('ai1-bias')
            self.metrics_data['ai1-scale'] = []
            self.metrics_data['ai1-bias'] = []

        # Add 'ai2-scale' and 'ai2-bias' if 'ai2' is in enabled_metrics
        if 'ai2' in enabled_metrics:
            self.enabled_metrics.append('ai2-scale')
            self.enabled_metrics.append('ai2-bias')
            self.metrics_data['ai2-scale'] = []
            self.metrics_data['ai2-bias'] = []

        if 'si' in enabled_metrics:
            self.enabled_metrics.append('si-scale')
            self.metrics_data['si-scale'] = []

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
            # Y = np.round(Y).astype(np.uint8)
            # Target = np.round(Target).astype(np.uint8)

            ai1, b1 = affine_invariant_1(Y, Target, confidence_map, irls_iters, eps)
            self.metrics_data['ai1'].append(ai1)
            self.metrics_data['ai1-scale'].append(b1[0])
            self.metrics_data['ai1-bias'].append(b1[1])
            return ai1, b1
        return None, None
    
    def affine_invariant_2(self, Y, Target, confidence_map=None, eps=1e-3):
        if 'ai2' in self.enabled_metrics:
            # Y = np.round(Y).astype(np.uint8)
            # Target = np.round(Target).astype(np.uint8)

            ai2, b2 = affine_invariant_2(Y, Target, confidence_map, eps)
            self.metrics_data['ai2'].append(ai2)
            self.metrics_data['ai2-scale'].append(b2[0])
            self.metrics_data['ai2-bias'].append(b2[1])
            return ai2, b2
        return None, None
    
    def scale_invariant(self, Y, Target, mask=None):
        if mask is None:
            mask = np.ones_like(Y)
        if 'si' in self.enabled_metrics:
            alpha = compute_scale(Y, Target, mask)
            si = ((Target - alpha * Y) ** 2).mean()
            self.metrics_data['si'].append(si)
            self.metrics_data['si-scale'].append(alpha)
            return si, alpha
        return None

    def spearman_correlation(self, Y, Target):
        if 'sc' in self.enabled_metrics:
            sc = 1 - np.abs(spearman_correlation(Y, Target))
            self.metrics_data['sc'].append(sc)
            return sc
        return None
    
    def epe_bad_pixel_metrics(self, Y, Target):
        result = []
        if any('epe_bad' in metric for metric in self.enabled_metrics):
            
            for metric in self.enabled_metrics:
                if metric.startswith('epe_bad'):
                    threshold = float('.'.join(metric.split('_')[2:]))
                    self.metrics_data[metric].append(self.bad_pixel_metric(Y, Target, threshold))
                    result.append(self.metrics_data[metric])

        return result

    def ai2_bad_pixel_metrics(self, Y, Target):
        result = []
        if any('ai2_bad' in metric for metric in self.enabled_metrics):
            ai2, b2 = self.affine_invariant_2(Y, Target)
            
            for metric in self.enabled_metrics:
                if metric.startswith('ai2_bad'):
                    threshold = float('.'.join(metric.split('_')[2:]))
                    self.metrics_data[metric].append(self.bad_pixel_metric(Y*b2[0] + b2[1], Target, threshold))
                    result.append(self.metrics_data[metric])

        return result
    
    def bad_pixel_metric(self, Y, Target, threshold):
        diff = np.abs(Y - Target)
        bad_pixels = np.sum(diff > threshold)
        total_pixels = np.prod(Y.shape)
        return bad_pixels / total_pixels

    def end_point_error(self, Y, Target):
        if 'epe' in self.enabled_metrics:
            epe = np.mean(np.abs(Y - Target))
            self.metrics_data['epe'].append(epe)
            return epe
        return None

    def root_mean_squared_error(self, Y, Target):
        if 'rmse' in self.enabled_metrics:
            rmse = np.sqrt(np.mean((Y - Target) ** 2))
            self.metrics_data['rmse'].append(rmse)
            return rmse
        return None

    def get_latest_metrics(self):
        latest_metrics = {metric: values[-1] for metric, values in self.metrics_data.items()}
        return latest_metrics

    def get_mean_metrics(self):
        mean_metrics = {metric: np.mean(values) for metric, values in self.metrics_data.items()}
        return mean_metrics
        
    def save_metrics(self):
        with open(self.save_path, "w") as f:
            header = "filename " + " ".join(self.enabled_metrics) + " color-range\n"
            f.write(header)
            for i, filename in enumerate(self.filenames):
                line = f"{filename} "
                for metric in self.enabled_metrics:
                    if metric in self.metrics_data:
                        line += f"{self.metrics_data[metric][i]:.3f} "
                line += f"{self.color_range[i][0]:.3f}-{self.color_range[i][1]:.3f}\n"
                f.write(line)
            
            # write mean
            mean_metrics = self.get_mean_metrics()
            mean_line = "mean "
            for metric in self.enabled_metrics:
                if metric in mean_metrics:
                    mean_line += f"{mean_metrics[metric]:.5f} "
            mean_line += "\n"
            f.write(mean_line)
            f.write("----end----\n")

"""
dir_path = '/mnt/d/Mono+Dual/QPDNet/result/eval/dp-disp'
gt_pattern = dir('/mnt/e/dual-pixel-dataset/MDD_dataset/test_c/target_depth/_npy_gt_832_1504/*.TIF')
"""


if __name__ == '__main__':
    # Test
    eval = Eval()
    eval.add_filename('test1.jpg')
    eval.add_filename('test2.jpg')
    eval.add_colorrange(0, 255)
    eval.add_colorrange(0, 255)

    Y = np.random.rand(1, 100, 100)
    Target = np.random.rand(1, 100, 100)
    eval.affine_invariant_1(Y, Target)
    eval.affine_invariant_2(Y, Target)
    eval.scale_invariant(Y, Target)
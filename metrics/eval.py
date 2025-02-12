from metrics.affine_invariant_metrics import *
import os.path as osp
from datetime import datetime
import numpy as np


class Eval():
    def __init__(self, save_path='', enabled_metrics=None):
        if enabled_metrics is None:
            enabled_metrics = []

        self.enabled_metrics = enabled_metrics
        
        self.metrics_data = {metric: [] for metric in enabled_metrics}

        # Add 'ai1-scale' and 'ai1-bias' if 'ai1' is in enabled_metrics
        if 'ai1' in enabled_metrics:
            self.metrics_data['ai1-scale'] = []
            self.metrics_data['ai1-bias'] = []

        # Add 'ai2-scale' and 'ai2-bias' if 'ai2' is in enabled_metrics
        if 'ai2' in enabled_metrics:
            self.metrics_data['ai2-scale'] = []
            self.metrics_data['ai2-bias'] = []

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
            ai1, b1 = affine_invariant_1(Y, Target, confidence_map, irls_iters, eps)
            self.metrics_data['ai1'].append(ai1)
            self.metrics_data['ai1-scale'].append(b1[0])
            self.metrics_data['ai1-bias'].append(b1[1])
            return ai1, b1
        return None, None
    
    def affine_invariant_2(self, Y, Target, confidence_map=None, eps=1e-3):
        if 'ai2' in self.enabled_metrics:
            ai2, b2 = affine_invariant_2(Y, Target, confidence_map, eps)
            self.metrics_data['ai2'].append(ai2)
            self.metrics_data['ai2-scale'].append(b2[0])
            self.metrics_data['ai2-bias'].append(b2[1])
            return ai2, b2
        return None, None

    def spearman_correlation(self, Y, Target):
        if 'sc' in self.enabled_metrics:
            sc = spearman_correlation(Y, Target)
            self.metrics_data['sc'].append(sc)
            return sc
        return None

    def ai2_bad_pixel_metrics(self, Y, Target):
        result = []
        if any('bad' in metric for metric in self.enabled_metrics):
            ai2, b2 = self.affine_invariant_2(Y, Target)
            if 'ai2_bad_0_005px' in self.enabled_metrics:
                self.metrics_data['ai2_bad_0_005px'].append(self.bad_pixel_metric(Y*b2[0] + b2[1], Target, 0.005))
                result.append(self.metrics_data['ai2_bad_0_005px'])
            if 'ai2_bad_0_01px' in self.enabled_metrics:
                self.metrics_data['ai2_bad_0_01px'].append(self.bad_pixel_metric(Y*b2[0] + b2[1], Target, 0.01))
                result.append(self.metrics_data['ai2_bad_0_01px'])
            if 'ai2_bad_0_05px' in self.enabled_metrics:
                self.metrics_data['ai2_bad_0_05px'].append(self.bad_pixel_metric(Y*b2[0] + b2[1], Target, 0.05))
                result.append(self.metrics_data['ai2_bad_0_05px'])
            if 'ai2_bad_0_1px' in self.enabled_metrics:
                self.metrics_data['ai2_bad_0_1px'].append(self.bad_pixel_metric(Y*b2[0] + b2[1], Target, 0.1))
                result.append(self.metrics_data['ai2_bad_0_1px'])
            if 'ai2_bad_0_5px' in self.enabled_metrics:
                self.metrics_data['ai2_bad_0_5px'].append(self.bad_pixel_metric(Y*b2[0] + b2[1], Target, 0.5))
                result.append(self.metrics_data['ai2_bad_0_5px'])
            if 'ai2_bad_1px' in self.enabled_metrics:
                self.metrics_data['ai2_bad_1px'].append(self.bad_pixel_metric(Y*b2[0] + b2[1], Target, 1.0))
                result.append(self.metrics_data['ai2_bad_1px'])
            if 'ai2_bad_3px' in self.enabled_metrics:
                self.metrics_data['ai2_bad_3px'].append(self.bad_pixel_metric(Y*b2[0] + b2[1], Target, 3.0))
                result.append(self.metrics_data['ai2_bad_3px'])
            if 'ai2_bad_5px' in self.enabled_metrics:
                self.metrics_data['ai2_bad_5px'].append(self.bad_pixel_metric(Y*b2[0] + b2[1], Target, 5.0))
                result.append(self.metrics_data['ai2_bad_5px'])
            if 'ai2_bad_10px' in self.enabled_metrics:
                self.metrics_data['ai2_bad_10px'].append(self.bad_pixel_metric(Y*b2[0] + b2[1], Target, 10.0))
                result.append(self.metrics_data['ai2_bad_10px'])
            if 'ai2_bad_15px' in self.enabled_metrics:
                self.metrics_data['ai2_bad_15px'].append(self.bad_pixel_metric(Y*b2[0] + b2[1], Target, 15.0))
                result.append(self.metrics_data['ai2_bad_15px'])
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
                    mean_line += f"{mean_metrics[metric]:.3f} "
            mean_line += "\n"
            f.write(mean_line)
            f.write("----end----\n")
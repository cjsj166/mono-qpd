from metrics.affine_invariant_metrics import *
import os.path as osp
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

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
        self.binned_epe_data = []  # Store binned EPE results for each sample
        self.bin_edges = None  # Store the bin edges used for binned EPE

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_path = save_path + '_eval_' + timestamp + '.txt'

    def add_colorrange(self, vmin, vmax):
        self.color_range.append((vmin, vmax))

    def add_filename(self, filename):
        self.filenames.append(filename)

    def add_binned_epe(self, epe_per_bin, pixel_count_per_bin, bin_edges=None):
        """Add binned EPE results for current sample"""
        self.binned_epe_data.append({
            'epe_per_bin': epe_per_bin,
            'pixel_count_per_bin': pixel_count_per_bin
        })
        
        # Store bin edges if provided (only need to store once)
        if bin_edges is not None and self.bin_edges is None:
            self.bin_edges = bin_edges

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

    def spearman_correlation(self, Y, Target, confidence_map=None):
        if 'sc' in self.enabled_metrics:
            sc = 1 - np.abs(spearman_correlation(Y, Target, W=confidence_map))
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

    def ai2_bad_pixel_metrics(self, Y, Target, confidence_map=None):
        result = []
        if any('ai2_bad' in metric for metric in self.enabled_metrics):
            ai2, b2 = self.affine_invariant_2(Y, Target)
            
            for metric in self.enabled_metrics:
                if metric.startswith('ai2_bad'):
                    threshold = float('.'.join(metric.split('_')[2:]))
                    self.metrics_data[metric].append(self.bad_pixel_metric(Y*b2[0] + b2[1], Target, threshold, confidence_map=confidence_map))
                    result.append(self.metrics_data[metric])

        return result
    
    def bad_pixel_metric(self, Y, Target, threshold, confidence_map=None):
        if confidence_map is None:
            confidence_map = np.ones_like(Target)
        conf = confidence_map.ravel()
        y = Y.ravel()
        t = Target.ravel()
        diff = np.abs(y - t)
        bad_pixels = np.sum((diff > threshold) * conf)
        total_pixels = np.sum(conf)
        if total_pixels == 0:
            return np.nan
        return bad_pixels / total_pixels

    def binned_epe(self, Y, Target, confidence_map=None, bins=range(-30, 31)):
        """
        Calculate EPE for each disparity bin.
        
        Args:
            Y: Predicted disparity (numpy array)
            Target: Ground truth disparity (numpy array) 
            confidence_map: Confidence mask (numpy array, optional)
            bins: Disparity bin edges (default: -30 to 30 with step 1)
            
        Returns:
            epe_per_bin: Dictionary with bin centers as keys and EPE values as values
            pixel_count_per_bin: Dictionary with bin centers as keys and pixel counts as values
        """
        if confidence_map is None:
            confidence_map = np.ones_like(Target)
            
        # Flatten arrays
        pred_flat = Y.ravel()
        gt_flat = Target.ravel() 
        conf_flat = confidence_map.ravel()
        
        # Apply confidence mask
        valid_mask = conf_flat > 0
        pred_valid = pred_flat[valid_mask]
        gt_valid = gt_flat[valid_mask]
        
        # Count pixels outside bin range for reporting
        # if hasattr(bins, '__len__') and len(bins) > 1:
        #     bin_min, bin_max = min(bins), max(bins)
        #     outside_mask = (gt_valid < bin_min) | (gt_valid >= bin_max)
        #     outside_count = np.sum(outside_mask)
        #     total_valid = len(gt_valid)
            # if outside_count > 0:
            #     print(f"  Note: {outside_count:,}/{total_valid:,} pixels ({outside_count/total_valid*100:.1f}%) outside bin range [{bin_min:.2f}, {bin_max:.2f}]")
        
        epe_per_bin = {}
        pixel_count_per_bin = {}
        
        # Calculate EPE for each bin
        for i in range(len(bins) - 1):
            bin_min = bins[i]
            bin_max = bins[i + 1]
            bin_center = (bin_min + bin_max) / 2.0
            
            # Find pixels in this disparity bin based on ground truth
            bin_mask = (gt_valid >= bin_min) & (gt_valid < bin_max)
            
            if np.sum(bin_mask) > 0:
                # Calculate EPE for pixels in this bin
                bin_pred = pred_valid[bin_mask]
                bin_gt = gt_valid[bin_mask]
                bin_epe = np.mean(np.abs(bin_pred - bin_gt))
                
                epe_per_bin[bin_center] = bin_epe
                pixel_count_per_bin[bin_center] = np.sum(bin_mask)
            else:
                epe_per_bin[bin_center] = np.nan
                pixel_count_per_bin[bin_center] = 0
                
        return epe_per_bin, pixel_count_per_bin


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
            header = "filename " + " ".join(self.enabled_metrics) + "\n"
            f.write(header)
            for i, filename in enumerate(self.filenames):
                line = f"{filename} "
                for metric in self.enabled_metrics:
                    if metric in self.metrics_data:
                        line += f"{self.metrics_data[metric][i]:.5f} "
                line += "\n"
                f.write(line)
            
            # write mean
            mean_metrics = self.get_mean_metrics()
            mean_line = "mean "
            for metric in self.enabled_metrics:
                if metric in mean_metrics:
                    mean_line += f"{mean_metrics[metric]:.5f} "
            mean_line += "\n"
            f.write(mean_line)

    def save_binned_epe(self):
        """Save binned EPE results to a separate file"""
        if not self.binned_epe_data:
            return
            
        binned_save_path = self.save_path.replace('.txt', '_binned_epe.txt')
        
        with open(binned_save_path, "w") as f:
            # Write header
            f.write("filename ")
            
            # Get all bin centers from first sample that has data
            bin_centers = None
            for data in self.binned_epe_data:
                if data['epe_per_bin']:
                    bin_centers = sorted(data['epe_per_bin'].keys())
                    break
                    
            if bin_centers is None:
                return
                
            # Write bin center headers
            for bin_center in bin_centers:
                f.write(f"bin_{bin_center:.1f}_epe bin_{bin_center:.1f}_count ")
            f.write("\n")
            
            # Write data for each sample
            for i, filename in enumerate(self.filenames):
                f.write(f"{filename} ")
                
                if i < len(self.binned_epe_data):
                    data = self.binned_epe_data[i]
                    for bin_center in bin_centers:
                        epe_val = data['epe_per_bin'].get(bin_center, np.nan)
                        count_val = data['pixel_count_per_bin'].get(bin_center, 0)
                        f.write(f"{epe_val:.5f} {count_val} ")
                else:
                    # Fill with NaN and 0 if no data
                    for bin_center in bin_centers:
                        f.write("nan 0 ")
                        
                f.write("\n")
            
            # Calculate and write mean values
            f.write("mean ")
            for bin_center in bin_centers:
                epe_values = []
                total_pixels = 0
                
                for data in self.binned_epe_data:
                    if bin_center in data['epe_per_bin'] and not np.isnan(data['epe_per_bin'][bin_center]):
                        epe_values.append(data['epe_per_bin'][bin_center])
                    if bin_center in data['pixel_count_per_bin']:
                        total_pixels += data['pixel_count_per_bin'][bin_center]
                
                mean_epe = np.mean(epe_values) if epe_values else np.nan
                f.write(f"{mean_epe:.5f} {total_pixels} ")
                
            f.write("\n")

    def plot_binned_epe_histogram(self):
        """Plot histogram of mean binned EPE values"""
        if not self.binned_epe_data:
            print("No binned EPE data available for plotting")
            return
            
        # Get all bin centers from first sample that has data
        bin_centers = None
        for data in self.binned_epe_data:
            if data['epe_per_bin']:
                bin_centers = sorted(data['epe_per_bin'].keys())
                break
                
        if bin_centers is None:
            print("No valid binned EPE data found")
            return
        
        # Calculate mean EPE and pixel counts for each bin
        bin_data = {}  # Use dict to handle duplicate bin centers
        
        for bin_center in bin_centers:
            epe_values = []
            total_pixels = 0
            
            for data in self.binned_epe_data:
                if bin_center in data['epe_per_bin'] and not np.isnan(data['epe_per_bin'][bin_center]):
                    epe_values.append(data['epe_per_bin'][bin_center])
                if bin_center in data['pixel_count_per_bin']:
                    total_pixels += data['pixel_count_per_bin'][bin_center]
            
            # Only include bins with at least 1 pixel and valid EPE
            if total_pixels >= 1 and epe_values:
                mean_epe = np.mean(epe_values)
                
                # Aggregate data for duplicate bin centers
                if bin_center in bin_data:
                    bin_data[bin_center]['epe_values'].extend(epe_values)
                    bin_data[bin_center]['total_pixels'] += total_pixels
                else:
                    bin_data[bin_center] = {
                        'epe_values': epe_values,
                        'total_pixels': total_pixels
                    }
        
        if not bin_data:
            print("No valid bins with pixel count >= 1 found")
            return
        
        # Prepare final data for plotting with bin ranges
        bin_ranges = []
        mean_epe_per_bin = []
        pixel_counts_per_bin = []
        bin_labels = []
        
        # Use stored bin edges if available, otherwise estimate from centers
        sorted_centers = sorted(bin_data.keys())
        
        if self.bin_edges is not None:
            # Use the actual bin edges that were used for binning
            actual_bin_edges = self.bin_edges
            print(f"Using actual bin edges: {actual_bin_edges}")
            print(f"Bin centers found: {sorted_centers}")
        else:
            # Fallback: estimate bin edges from centers (old method)
            print("Warning: No bin edges stored, estimating from centers")
            if len(sorted_centers) >= 2:
                actual_bin_edges = []
                
                # For the first bin, assume it starts from a reasonable minimum
                first_gap = sorted_centers[1] - sorted_centers[0]
                actual_bin_edges.append(sorted_centers[0] - first_gap/2)
                
                # For middle bins, use midpoints
                for i in range(len(sorted_centers) - 1):
                    midpoint = (sorted_centers[i] + sorted_centers[i+1]) / 2
                    actual_bin_edges.append(midpoint)
                
                # For the last bin, assume it ends at a reasonable maximum  
                last_gap = sorted_centers[-1] - sorted_centers[-2]
                actual_bin_edges.append(sorted_centers[-1] + last_gap/2)
            else:
                # Single bin case
                actual_bin_edges = [sorted_centers[0] - 0.1, sorted_centers[0] + 0.1]
        
        # Create bin labels with ranges using actual bin edges
        if self.bin_edges is not None and len(actual_bin_edges) > 1:
            # Direct approach: use the stored bin edges in order
            sorted_centers = sorted(bin_data.keys())
            
            # Create labels directly from the bin edges
            for i in range(len(actual_bin_edges) - 1):
                expected_center = (actual_bin_edges[i] + actual_bin_edges[i + 1]) / 2.0
                
                # Find the bin data that corresponds to this expected center
                closest_center = None
                min_diff = float('inf')
                for center in sorted_centers:
                    diff = abs(center - expected_center)
                    if diff < min_diff:
                        min_diff = diff
                        closest_center = center
                
                # Only include if we found a close match and it has data
                if closest_center is not None and min_diff < 0.1 and closest_center in bin_data:
                    data = bin_data[closest_center]
                    if data['total_pixels'] > 0:  # Only include bins with data
                        mean_epe = np.mean(data['epe_values'])
                        
                        bin_start = actual_bin_edges[i]
                        bin_end = actual_bin_edges[i + 1]
                        
                        # Format the range to show exact quantile values
                        # Use adaptive precision to show exact values from quantile_edges
                        def format_value(val):
                            # Check if it's a "nice" value that should be shown with fewer decimals
                            if abs(val - round(val, 0)) < 1e-10:  # Integer
                                return f"{val:.1f}"
                            elif abs(val - round(val, 1)) < 1e-10:  # One decimal
                                return f"{val:.1f}"
                            elif abs(val - round(val, 2)) < 1e-10:  # Two decimals
                                return f"{val:.2f}"
                            elif abs(val - round(val, 3)) < 1e-10:  # Three decimals
                                return f"{val:.3f}"
                            elif abs(val - round(val, 4)) < 1e-10:  # Four decimals
                                return f"{val:.4f}"
                            else:
                                return f"{val:.5f}"  # Default precision
                        
                        start_str = format_value(bin_start)
                        end_str = format_value(bin_end)
                        
                        if i == len(actual_bin_edges) - 2:  # Last bin includes end
                            label = f"[{start_str}, {end_str}]"
                        else:
                            label = f"[{start_str}, {end_str})"
                            
                        bin_labels.append(label)
                        bin_ranges.append(len(bin_ranges))  # Use index for x-axis positioning
                        mean_epe_per_bin.append(mean_epe)
                        pixel_counts_per_bin.append(data['total_pixels'])
        else:
            # Fallback to old method if no bin edges available
            for i, bin_center in enumerate(sorted_centers):
                data = bin_data[bin_center]
                mean_epe = np.mean(data['epe_values'])
                
                # Find the corresponding bin in actual_bin_edges by finding the closest center match
                best_match_idx = None
                min_diff = float('inf')
                
                for j in range(len(actual_bin_edges) - 1):
                    expected_center = (actual_bin_edges[j] + actual_bin_edges[j + 1]) / 2.0
                    diff = abs(expected_center - bin_center)
                    if diff < min_diff:
                        min_diff = diff
                        best_match_idx = j
                
                if best_match_idx is not None:
                    bin_start = actual_bin_edges[best_match_idx]
                    bin_end = actual_bin_edges[best_match_idx + 1]
                    
                    # Format the range to show exact quantile values
                    def format_value(val):
                        if abs(val - round(val, 0)) < 1e-10:  # Integer
                            return f"{val:.1f}"
                        elif abs(val - round(val, 1)) < 1e-10:  # One decimal
                            return f"{val:.1f}"
                        elif abs(val - round(val, 2)) < 1e-10:  # Two decimals
                            return f"{val:.2f}"
                        elif abs(val - round(val, 3)) < 1e-10:  # Three decimals
                            return f"{val:.3f}"
                        elif abs(val - round(val, 4)) < 1e-10:  # Four decimals
                            return f"{val:.4f}"
                        else:
                            return f"{val:.5f}"
                    
                    start_str = format_value(bin_start)
                    end_str = format_value(bin_end)
                    
                    if best_match_idx == len(actual_bin_edges) - 2:  # Last bin includes end
                        label = f"[{start_str}, {end_str}]"
                    else:
                        label = f"[{start_str}, {end_str})"
                        
                    bin_labels.append(label)
                    bin_ranges.append(len(bin_ranges))  # Use index for x-axis positioning
                    mean_epe_per_bin.append(mean_epe)
                    pixel_counts_per_bin.append(data['total_pixels'])
                else:
                    print(f"Warning: Could not match bin center {bin_center} to any edge")
        
        # Create histogram plot
        plt.figure(figsize=(16, 8))
        
        # Create bar plot with indices as x positions
        bars = plt.bar(bin_ranges, mean_epe_per_bin, width=0.6, 
                      alpha=0.7, color='skyblue', edgecolor='navy')
        
        # Add value labels on top of bars
        max_epe = max(mean_epe_per_bin)
        for i, (epe_val, pixel_count) in enumerate(zip(mean_epe_per_bin, pixel_counts_per_bin)):
            plt.text(i, epe_val + max_epe * 0.02, 
                    f'{epe_val:.3f}\n({pixel_count:,}px)', 
                    ha='center', va='bottom', fontsize=10, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Set plot properties
        plt.xlabel('Disparity Bin Range', fontsize=14)
        plt.ylabel('Mean EPE', fontsize=14)
        plt.title('Mean End-Point Error by Disparity Bin (Quantile-based)', fontsize=16)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Set x-axis labels and ticks
        plt.xticks(bin_ranges, bin_labels, rotation=45, ha='right')
        plt.xlim(-0.5, len(bin_ranges) - 0.5)
        
        # Add statistics text
        total_pixels = sum(pixel_counts_per_bin)
        mean_pixels_per_bin = total_pixels / len(bin_ranges)
        stats_text = f'Total bins: {len(bin_ranges)}\nTotal pixels: {total_pixels:,}\nMean pixels/bin: {mean_pixels_per_bin:,.0f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # Improve layout
        plt.tight_layout()
        
        # Save plot
        plot_save_path = self.save_path.replace('.txt', '_binned_epe_histogram.png')
        plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Binned EPE histogram saved to: {plot_save_path}")
        print(f"Valid disparity range: {min(sorted_centers):.3f} to {max(sorted_centers):.3f}")
        print(f"Total bins with data: {len(bin_ranges)}")
        print(f"Bin ranges: {bin_labels}")
        print(f"Pixel distribution: min={min(pixel_counts_per_bin):,}, max={max(pixel_counts_per_bin):,}, mean={mean_pixels_per_bin:,.0f}")

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
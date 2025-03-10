from glob import glob
from PIL import Image
import numpy as np
from metrics.eval import Eval
import os

def read_gt(path):
    if path.endswith('.npy'):
        gt = np.load(path)
    elif path.endswith('.TIF'):    
        gt = Image.open(path)
        gt = np.array(gt)
    return gt

gt_npy_pattern = '/mnt/e/dual-pixel-dataset/MDD_dataset/test_c/target_depth/_npy_gt_832_1504/*.TIF'
gt_tif_pattern = '/mnt/e/dual-pixel-dataset/MDD_dataset/test_c/target_depth/_npy_gt_832_1504/*.TIF'
est_pattern = '/mnt/d/Mono+Dual/QPDNet/result/eval/dp-disp/**/*.npy'

if __name__ == '__main__':

    gt_pattern = gt_npy_pattern
    est_paths = glob(est_pattern, recursive=True)
    gt_paths = glob(gt_pattern)

    save_dir = 'eval_test'
    os.makedirs(save_dir, exist_ok=True)

    eval_est = Eval(os.path.join(save_dir, 'eval'), enabled_metrics=['ai1', 'ai2', 'ai2_bad_1px', 'ai2_bad_3px', 'ai2_bad_5px', 'ai2_bad_10px', 'ai2_bad_15px'])
    
    for est_path, gt_path in zip(est_paths, gt_paths):
        est = np.load(est_path)
        est = est.squeeze(0)
        gt = read_gt(gt_path)
        gt = gt.astype(np.float32)

        gt = -gt / 255

        # print(type(est), est.dtype, est.shape, est.mean(), est.std())
        # print(type(gt), gt.dtype, gt.shape, gt.mean(), gt.std())

        ai1, b1 = eval_est.affine_invariant_1(est, gt)
        ai2, b2 = eval_est.affine_invariant_2(est, gt)

        print(ai1, b1[0], b1[1]) 
        print(ai2, b2[0], b2[1])

        pass


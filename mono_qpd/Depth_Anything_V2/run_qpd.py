import argparse
import cv2
import glob

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import os
import torch
from torchvision.transforms import Compose

from tqdm import tqdm
from depth_anything_v2.dpt import DepthAnythingV2
from dataloaders import Quad_datasets as datasets
from metrics.affine_invariant_metrics import affine_invariant_1, affine_invariant_2
import matplotlib.pyplot as plt
from depth_anything_v2.util.transform import Resize
from metrics.eval import Eval


from collections import OrderedDict

def fix_key(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    return new_state_dict

# Evaluate depth anything and qpdnet result.
# DepthAnything model is loaded and executed. QPDNet model itself is not executed in this code. 
# But the estimated result is loaded from the dp_src path.

@torch.no_grad()
def validate_MDD(depth_anything, args):

    gt_depth_dir = os.path.join(args.log_dir, 'depth_gt')
    center_dir = os.path.join(args.log_dir, 'center_est')
    dp_dir = os.path.join(args.log_dir, 'dp_est')
    center_img_dir = os.path.join(args.log_dir, 'center_img')
    os.makedirs(center_dir, exist_ok=True)
    os.makedirs(dp_dir, exist_ok=True)
    os.makedirs(center_img_dir, exist_ok=True)

    val_dataset = datasets.MDD(None, image_set='test', root=args.dataset_path, resize_size = 768)

    eval_center = Eval(os.path.join(args.log_dir, 'center'))
    eval_dp = Eval(os.path.join(args.log_dir, 'dp'))

    def stats(data):
        return data.mean(), data.std(), data.min(), data.max()
    
    # width_border = 50
    # height_border = 40

    # width_border = 100
    # height_border = 80

    for val_id in tqdm(range(len(val_dataset))):
        # result = self.image_list[index] + [self.disparity_list[index]], center_img, lrtb_list, depth
        paths, center_img, lrtb, flow_gt, valid_gt, center_ori = val_dataset[val_id]
        center_img = center_img.to(DEVICE).float()
        flow_gt = flow_gt.cpu().numpy()
        center_ori = center_ori.permute(1, 2, 0).cpu().numpy()

        center_pred = depth_anything(center_img[None])[0]
        center_pred = center_pred.cpu().numpy()
        
        center_img = center_img.permute(1, 2, 0).cpu().numpy()

        # Make directories to save the results ex) center_est/0001/0001.jpg
        pth_lists = paths[0].split('/')[-3:]
        pth = '/'.join(pth_lists)
        
        os.makedirs(os.path.join(center_dir, os.path.dirname(pth)), exist_ok=True)
        os.makedirs(os.path.join(center_img_dir, os.path.dirname(pth)), exist_ok=True)
        os.makedirs(os.path.join(gt_depth_dir, os.path.dirname(pth)), exist_ok=True)

        # Calculate affine invariant metrics
        eval_center.add_filename(os.path.basename(pth))
        center_ai1, center_b1 = eval_center.affine_invariant_1(center_pred, flow_gt)
        center_ai2, center_b2 = eval_center.affine_invariant_2(center_pred, flow_gt)
        eval_center.spearman_correlation(center_pred, flow_gt)

        eval_dp.add_filename(os.path.basename(pth))
        # dp_ai1, dp_b1 = eval_dp.affine_invariant_1(dp_pred, flow_gt)
        # dp_ai2, dp_b2 = eval_dp.affine_invariant_2(dp_pred, flow_gt)
        # eval_dp.spearman_correlation(-dp_pred, flow_gt)

        # Get fit predicted disparity
        center_ai2_fit = center_pred * center_b2[0] + center_b2[1]
        # dp_ai2_fit = dp_pred * dp_b2[0] + dp_b2[1]

        # Compute min and max value for colormap
        vmin, vmax = np.min([center_ai2_fit, flow_gt]), np.max([center_ai2_fit, flow_gt])
        eval_center.add_colorrange(vmin, vmax)
        eval_dp.add_colorrange(vmin, vmax)

        # Save in colormap
        plt.imsave(os.path.join(center_dir, pth), center_ai2_fit.squeeze(), cmap='jet', vmin=vmin, vmax=vmax)        
        plt.imsave(os.path.join(center_dir, pth.replace('.jpg', '_err.jpg')), center_ai2_fit.squeeze() - flow_gt.squeeze(), cmap='jet')
        
        plt.imsave(os.path.join(gt_depth_dir, pth), flow_gt.squeeze(), cmap='jet', vmin=vmin, vmax=vmax)
        plt.imsave(os.path.join(center_img_dir, pth), center_ori.astype(np.uint8))

        np.save(os.path.join(center_dir, pth.replace('.jpg', '.npy')), center_pred.squeeze())
        # np.save(os.path.join(dp_dir, pth.replace('.jpg', '.npy')), dp_pred.squeeze())


    # Save metrics in txt file
    eval_center.save_metrics()
    eval_dp.save_metrics()

    bar = np.array(range(300)).reshape(1, 300)
    bar = np.repeat(bar, 20, axis=0)

    plt.imsave(os.path.join(args.log_dir, 'colorbar.png'), bar, cmap='jet')

@torch.no_grad()
def validate_Real_QPD(depth_anything, args):
    dp_src = args.dp_src

    center_dir = os.path.join(args.log_dir, 'center_est')
    dp_dir = os.path.join(args.log_dir, 'dp_est')
    center_img_dir = os.path.join(args.log_dir, 'center_img')
    os.makedirs(center_dir, exist_ok=True)
    os.makedirs(dp_dir, exist_ok=True)
    os.makedirs(center_img_dir, exist_ok=True)

    val_dataset = datasets.Real_QPD(None, image_set='test', root=args.dataset_path, no_gt_disp=True, no_gt_aif=True)
    # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)

    # gt_resize = Resize(
    #         width=768,
    #         height=768,
    #         resize_target=False,
    #         keep_aspect_ratio=True,
    #         # ensure_multiple_of=14,
    #         resize_method='lower_bound',
    #         image_interpolation_method=cv2.INTER_LINEAR
    #         )

    # for i, sample in tqdm(enumerate(val_dataset), total=len(val_dataset)):  # 각 샘플을 안전하게 로드
    #     print(f"Sample {i}:")

    # with open(os.path.join(args.log_dir, 'est.txt'), 'a') as f:
    #     f.write(f'path aif_ai2 center_ai2 dp_ai2 vmin vmax \n')

    for val_id in tqdm(range(len(val_dataset))):
        
        # paths, center, lrtb, aif, flow_gt, valid_gt = batch
        paths, center, lrtb = val_dataset[val_id]

        center = center.permute(1, 2, 0).cpu().numpy()

        _ , center_pred = depth_anything.infer_image(center, 768)
        
        # center_img = gt_resize({'image': center})['image']
        center_img = center

        # image = transform({'image': image})['image']

        # center_pred = gt_resize({'image': center_pred})['image']
        
        
        # paths = [pth.split('/')[-5:] for pth in paths]
        pth_lists = paths[0].split('/')[-3:]
        pth = '/'.join(pth_lists)
        
        os.makedirs(os.path.join(center_dir, os.path.dirname(pth)), exist_ok=True)
        os.makedirs(os.path.join(center_img_dir, os.path.dirname(pth)), exist_ok=True)

        dp_pth = pth.replace('.png', '.npy')
        dp_pred = np.load(os.path.join(dp_src, dp_pth))

        # vmin, vmax = np.min([center_pred, dp_pred]), np.max([center_pred, dp_pred])
        center_pred_min = np.min(center_pred)
        center_pred_max = np.max(center_pred)
        dp_pred_min = np.min(dp_pred)
        dp_pred_max = np.max(dp_pred)

        plt.imsave(os.path.join(center_dir, pth), center_pred.squeeze(), cmap='jet', vmin=center_pred_min, vmax=center_pred_max)
        plt.imsave(os.path.join(dp_dir, pth), dp_pred.squeeze(), cmap='jet', vmin=dp_pred_min, vmax=dp_pred_max)

        plt.imsave(os.path.join(center_img_dir, pth), center_img.astype(np.uint8))
            
        if val_id == len(val_dataset):
            break

    bar = np.array(range(300)).reshape(1, 300)
    bar = np.repeat(bar, 20, axis=0)

    plt.imsave(os.path.join(args.log_dir, 'colorbar.png'), bar, cmap='jet')


@torch.no_grad()
def validate_QPD(depth_anything, args):
    aif_dir = os.path.join(args.log_dir, 'aif_est')
    center_dir = os.path.join(args.log_dir, 'center_est')
    aif_img_dir = os.path.join(args.log_dir, 'aif_img')
    dp_dir = os.path.join(args.log_dir, 'dp_est')
    center_img_dir = os.path.join(args.log_dir, 'center_img')
    gt_dir = os.path.join(args.log_dir, 'gt')
    os.makedirs(center_dir, exist_ok=True)
    os.makedirs(dp_dir, exist_ok=True)
    os.makedirs(center_img_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    val_dataset = datasets.QPD(None, image_set='test', root=args.dataset_path)

    gt_resize = Resize(
            width=768,
            height=768,
            resize_target=False,
            keep_aspect_ratio=True,
            # ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_LINEAR
            )


    for val_id in tqdm(range(len(val_dataset))):

        # result = self.image_list[index] + [self.disparity_list[index]], center_img, lrtb_list, flow, valid, center_ori
        paths, center, lrtb, flow_gt, valid_gt, center_ori = val_dataset[val_id]

        center = center.to(DEVICE).float()

        # aif = aif.permute(1, 2, 0).cpu().numpy()

        center_pred = depth_anything(center[None])[0]

        center_pred = center_pred.cpu().numpy()
        # # aif_pred = depth_anything(aif)
        
        # center_img = gt_resize({'image': center})['image']
        # aif_img = gt_resize({'image': aif})['image']

        # center_pred = gt_resize({'image': center_pred})['image']
        # aif_pred = gt_resize({'image': aif_pred})['image']

        center = center.permute(1, 2, 0).cpu().numpy()
        flow_gt = flow_gt.cpu().numpy()
        
        pth_lists = paths[0].split('/')[-3:]
        pth = '/'.join(pth_lists)
        pth_lists[-1] = pth_lists[-1].replace('.png', '_-10_3_range.png')
        fixed_range_result_pth = '/'.join(pth_lists)

        # os.makedirs(os.path.join(aif_dir, os.path.dirname(pth)), exist_ok=True)
        os.makedirs(os.path.join(center_dir, os.path.dirname(pth)), exist_ok=True)
        # os.makedirs(os.path.join(aif_img_dir, os.path.dirname(pth)), exist_ok=True)
        os.makedirs(os.path.join(center_img_dir, os.path.dirname(pth)), exist_ok=True)
        os.makedirs(os.path.join(gt_dir, os.path.dirname(pth)), exist_ok=True)

        # # # aif_ai2, aif_b = affine_invariant_2(aif_pred, flow_gt)
        center_ai2, center_b = affine_invariant_2(center_pred, flow_gt)

        # # # # fit_aif_pred = aif_pred * aif_b[0] + aif_b[1]
        fit_center_pred = center_pred * center_b[0] + center_b[1]
        # vmin, vmax = np.min([fit_aif_pred, fit_center_pred, flow_gt]), np.max([fit_aif_pred, fit_center_pred, flow_gt])
        vmin, vmax = np.min([fit_center_pred, flow_gt]), np.max([fit_center_pred, flow_gt])

        # # plt.imsave(os.path.join(aif_dir, pth), fit_aif_pred.squeeze(), cmap='jet', vmin=vmin, vmax=vmax) # without vmin and vmax, colormap will be min-max normalized.
        plt.imsave(os.path.join(center_dir, pth), fit_center_pred.squeeze(), cmap='jet', vmin=vmin, vmax=vmax)
        plt.imsave(os.path.join(gt_dir, pth), flow_gt, cmap='jet', vmin=vmin, vmax=vmax)

        # # plt.imsave(os.path.join(aif_dir, fixed_range_result_pth), fit_aif_pred.squeeze(), cmap='jet', vmin=-10, vmax=3)
        plt.imsave(os.path.join(center_dir, fixed_range_result_pth), fit_center_pred.squeeze(), cmap='jet', vmin=-10, vmax=3)
        plt.imsave(os.path.join(gt_dir, fixed_range_result_pth), flow_gt, cmap='jet', vmin=-10, vmax=3)

        # plt.imsave(os.path.join(aif_img_dir, pth), aif_img.astype(np.uint8))
        # plt.imsave(os.path.join(center_img_dir, pth), center_img.astype(np.uint8))
    
        with open(os.path.join(args.log_dir, 'est.txt'), 'w') as f:
            print(f'{pth} {center_ai2:0.3f} {vmin:0.3f} {vmax:0.3f}\n')
            f.write(f'{pth} {center_ai2:0.3f} {vmin:0.3f} {vmax:0.3f}\n')
        
        if val_id == len(val_dataset):
            break

    bar = np.array(range(300)).reshape(1, 300)
    bar = np.repeat(bar, 20, axis=0)

    plt.imsave(os.path.join(args.log_dir, 'colorbar.png'), bar, cmap='jet')



if __name__ == '__main__':    
    RANDOM_SEED = 100
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Create a general generator for use with the test dataloader
    general_generator = torch.Generator()
    general_generator.manual_seed(RANDOM_SEED)


    parser = argparse.ArgumentParser(description='Depth Anything V2')

    parser.add_argument('--dataset', help="dataset for evaluation", required=False, choices=["QPD", "Real_QPD", "MDD"], default="QPD")
    parser.add_argument('--dataset_path', type=str, default='', required=True)
    parser.add_argument('--checkpoint', type=str, default='')
    # parser.add_argument('--dp_src', type=str, default='', required=True)

    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--log_dir', type=str, default='./result')
    parser.add_argument('--test_type', type=str, default='center')

    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])


    # parser.add_argument('--pred_only', dest='pred_only', action='store_true', help='only display the prediction')
    # parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

    args = parser.parse_args()

    # dp_src = '/mnt/d/Mono+Dual/QPDNet/res/dp_est'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])

    if args.checkpoint:
        d = torch.load(f'{args.checkpoint}', map_location='cpu')
        d['model'] = fix_key(d['model'])
        depth_anything.load_state_dict(d['model'])
    else:
        depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    
    depth_anything = depth_anything.to(DEVICE).eval()

    if args.dataset == "QPD":
        validate_QPD(depth_anything, args)

    if args.dataset == "Real_QPD":
        validate_Real_QPD(depth_anything, args)
    
    if args.dataset == "MDD":
        validate_MDD(depth_anything, args)

    
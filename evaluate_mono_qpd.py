from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from mono_qpd.QPDNet.qpd_net import QPDNet, autocast
import mono_qpd.QPDNet.Quad_datasets as datasets
from mono_qpd.QPDNet.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
import os.path as osp
import os
import cv2
from mono_qpd.mono_qpd import MonoQPD
from argparse import Namespace
import torch.nn as nn
from mono_qpd.loss import LeastSquareScaleInvariantLoss
from matplotlib import cm
import torch.utils.data as data

from metrics.eval import Eval
from collections import OrderedDict

def fix_key(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    return new_state_dict

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_colormap(depth_range, dpi):
    ##setting for colormap
    diff = depth_range[1] - depth_range[0]
    cm = plt.get_cmap('jet', diff * dpi)
    delta = diff / cm.N
    value = np.arange(depth_range[0], depth_range[1], delta)
    norm = BoundaryNorm(value, ncolors=cm.N)
    norm.clip = False
    cm.set_under('gray')
    return cm, norm

def show_colormap(value, path, depth_range, dpi, figsize=(12, 10)):
    ##color map setting
    cm, norm = set_colormap(depth_range, dpi)

    ##plot color map
    plt.figure(figsize=figsize)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.imshow(value, cmap=cm, norm=norm)
    plt.colorbar(orientation='vertical')

    ##show or save map
    if (len(path) > 0):
        folder = osp.dirname(path)
        if not osp.exists(folder):
            os.makedirs(folder)
        plt.savefig(path)
    else:
        plt.show()

    ##close plot
    plt.clf()

def save_image(value, path, cmap='jet', vmin=None, vmax=None):
    """
    Save an image with matplotlib's imsave, using specified colormap and value range.
    """
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    plt.imsave(path, value, cmap=cmap, vmin=vmin, vmax=vmax)


@torch.no_grad()
def validate_Real_QPD(model, datatype='dual', iters=32, mixed_prec=False, save_result=False, val_save_skip=1, image_set='test', path='', save_path='', batch_size=31, preprocess_params={'crop_h':1052, 'crop_w':1315, 'resize_h': 896, 'resize_w':1120}):
    model.eval()
    aug_params = {}
    
    if path == '':
        val_dataset = datasets.Real_QPD(datatype=datatype, aug_params=aug_params, image_set=image_set, preprocess_params=preprocess_params)
    else:
        val_dataset = datasets.Real_QPD(datatype=datatype, aug_params=aug_params, image_set=image_set, preprocess_params=preprocess_params, root=path)

    # TODO : revert worker number
    # val_loader = data.DataLoader(val_dataset, batch_size=batch_size, 
    #     pin_memory=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, drop_last=False)
    
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, 
        pin_memory=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, drop_last=False)
    
    est_dir = os.path.join(save_path, 'est')
    vminvmax_dir = os.path.join(save_path, 'vminvmax')
    src_dir = os.path.join(save_path, 'src')
    os.makedirs(est_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)

    path = os.path.basename(os.path.dirname(path))

    # ai2_bad_0_005px ~ ai2_bad_15px
    eval_est = Eval(os.path.join(save_path, 'center'), enabled_metrics=['epe', 'rmse', 'ai1', 'ai2', 'si', 'epe_bad_0_005px', 'epe_bad_0_01px', 'epe_bad_0_05px', 'epe_bad_0_1px', 'epe_bad_0_5px', 'epe_bad_1px'])
    
    result = {}

    if val_save_skip < batch_size:
        val_save_skip = 1
    else:
        val_save_skip = val_save_skip // batch_size

    # for val_id in tqdm(range(val_num)):
    for i_batch, data_blob in enumerate(tqdm(val_loader)):

        if i_batch % val_save_skip != 0:
            continue
        # if val_id == 2:
        #     break
        # paths, image1, image2, flow_gt, valid_gt = data_blob

        image_paths = data_blob['image_list']
        center = data_blob['center'].cuda()
        lrtb_list = data_blob['lrtb_list'].cuda()
        
        concat_lr = torch.cat([lrtb_list[:,0],lrtb_list[:,1]], dim=0).contiguous()
        
        with autocast(enabled=mixed_prec):
            _, flow_pr = model(center, concat_lr, iters=iters, test_mode=True)


        # flow_pr = torch.zeros_like(flow_gt)

        # Align dimensions and file format
        flow_pr = flow_pr.cpu().numpy()
        center = center.permute(0,2,3,1).cpu().numpy()

        # assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

        for i in range(batch_size):
            flow_pr_i = flow_pr[i]
            center_i = center[i]

            if not os.path.exists('result/predictions/'+path+'/'):
                os.makedirs('result/predictions/'+path+'/')
            
            pth = image_paths[0][i].split('/')[-6:]
            pth = '/'.join(pth)

            os.makedirs(os.path.join(est_dir, os.path.dirname(pth)), exist_ok=True)
            os.makedirs(os.path.join(vminvmax_dir, os.path.dirname(pth)), exist_ok=True)
            # flow_prn = flow_pr.cpu().numpy().squeeze()

            os.makedirs(os.path.join(src_dir, os.path.dirname(pth)), exist_ok=True)
            os.makedirs(os.path.join(src_dir, os.path.dirname(pth)), exist_ok=True)
            os.makedirs(os.path.join(src_dir, os.path.dirname(pth)), exist_ok=True)

            plt.imsave(os.path.join(src_dir, pth), center_i.astype(np.uint8))
            # plt.imsave(os.path.join(src_dir, pth), image2[0].astype(np.uint8))
            # plt.imsave(os.path.join(src_dir, pth), image2[1].astype(np.uint8))

            # print(flow_pr_i.min(), flow_pr_i.max())
            vmin, vmax = -4, 1.5
            plt.imsave(os.path.join(vminvmax_dir, pth), flow_pr_i.squeeze(), cmap='jet', vmin=vmin, vmax=vmax)
            plt.imsave(os.path.join(est_dir, pth), flow_pr_i.squeeze(), cmap='jet')

    return None

@torch.no_grad()
def validate_DPD_Disp(model, datatype='dual', gt_types=['inv_depth'], iters=32, mixed_prec=False, save_result=False, val_save_skip=1, image_set='test', path='', save_path='result/predictions', batch_size=1, preprocess_params={'crop_h':2940, 'crop_w':5145, 'resize_h': 224*4, 'resize_w':224*7}):
    """ Peform validation using the FlyingThings3D (TEST) split """
    model.eval()
    aug_params = {}
    
    if path == '':
        val_dataset = datasets.DPD_Disp(datatype=datatype, gt_types=gt_types, aug_params=aug_params, preprocess_params=preprocess_params, image_set=image_set)
    else:
        val_dataset = datasets.DPD_Disp(datatype=datatype, gt_types=gt_types, aug_params=aug_params, image_set=image_set, preprocess_params=preprocess_params, root=path)

    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, 
        pin_memory=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, drop_last=False)    

    npy_dir = os.path.join(save_path, 'npy')
    est_dir = os.path.join(save_path, 'est')
    err_dir = os.path.join(save_path, 'err')
    gt_dir = os.path.join(save_path, 'gt')
    src_dir = os.path.join(save_path, 'src')
    src_test_c_dir = os.path.join(src_dir, 'test_c', 'source', 'scenes')
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(est_dir, exist_ok=True)
    os.makedirs(err_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(src_test_c_dir, exist_ok=True)

    eval_est = Eval(os.path.join(save_path, 'center'), enabled_metrics=['ai1', 'ai2', 'ai2_bad_0_003', 'ai2_bad_0_005', 'ai2_bad_0_01', 'ai2_bad_0_03', 'ai2_bad_0_05'])

    result = {}

    if val_save_skip < batch_size:
        val_save_skip = 1
    else:
        val_save_skip = val_save_skip // batch_size

    # for val_id in tqdm(range(val_num)):
    for i_batch, data_blob in enumerate(tqdm(val_loader)):

        if i_batch % val_save_skip != 0:
            continue

        image_paths = data_blob['image_list']
        center = data_blob['center'].cuda()
        lrtb_list = data_blob['lrtb_list'].cuda()
        inv_depth_gt =  data_blob['inv_depth'].cuda()
        valid_gt = data_blob['inv_depth_valid'].cuda()

        concat_lr = torch.cat([lrtb_list[:,0],lrtb_list[:,1]], dim=0).contiguous()
        
        with autocast(enabled=mixed_prec):
            _, flow_pr = model(center, concat_lr, iters=iters, test_mode=True)

        # Crop invalid regions
        h, w = flow_pr.shape[-2:]
        flow_pr = flow_pr[..., 32:h-32, 32:w-32]
        inv_depth_gt = inv_depth_gt[..., 32:h-32, 32:w-32]

        # flow_pr = torch.zeros_like(flow_gt)

        # Align dimensions and file format
        flow_pr = flow_pr.cpu().numpy()
        inv_depth_gt = inv_depth_gt.cpu().numpy()
        center = center.permute(0,2,3,1).cpu().numpy()
        
        assert flow_pr.shape == inv_depth_gt.shape, (flow_pr.shape, inv_depth_gt.shape)

        for i in range(batch_size):
            flow_pr_i = flow_pr[i]
            inv_depth_gt_i = inv_depth_gt[i]
            center_i = center[i]
            est_ai1, est_b1 = eval_est.affine_invariant_1(flow_pr_i, inv_depth_gt_i)
            est_ai2, est_b2 = eval_est.affine_invariant_2(flow_pr_i, inv_depth_gt_i)
            bads = eval_est.ai2_bad_pixel_metrics(flow_pr_i, inv_depth_gt_i)
            est_ai2_fit = flow_pr_i * est_b2[0] + est_b2[1]

            # print(flow_pr_i[0][418:421][317:343])

            if save_result:
                if not os.path.exists('result/predictions/'+path+'/'):
                    os.makedirs('result/predictions/'+path+'/')
                
                pth_lists = image_paths[0][i].split('/')[-3:]
                pth = '/'.join(pth_lists)
                pth = os.path.basename(pth)

                # Set range
                vmargin = 0.3
                vrng = inv_depth_gt_i.max() - inv_depth_gt_i.min()
                vmin, vmax = inv_depth_gt_i.min() - vrng * vmargin, inv_depth_gt_i.max() + vrng * vmargin
                vmin_err, vmax_err = 0, vrng * vmargin

                eval_est.add_colorrange(vmin, vmax)

                # Save as npy
                np.save(os.path.join(npy_dir, pth.replace('.jpg', '.npy')), flow_pr_i)

                npy_gt_dir = os.path.join(save_path, 'npy_gt')
                os.makedirs(npy_gt_dir, exist_ok=True)
                np.save(os.path.join(npy_gt_dir, pth.replace('.jpg', '.npy')), inv_depth_gt_i)

                # Save in colormap
                plt.imsave(os.path.join(est_dir, pth), est_ai2_fit.squeeze(), cmap='jet', vmin=vmin, vmax=vmax)
                plt.imsave(os.path.join(err_dir, pth), np.abs(est_ai2_fit.squeeze() - inv_depth_gt_i.squeeze()), cmap='jet', vmin=vmin_err, vmax=vmax_err)
                
                plt.imsave(os.path.join(gt_dir, pth), inv_depth_gt_i.squeeze(), cmap='jet', vmin=vmin, vmax=vmax)

                plt.imsave(os.path.join(src_test_c_dir, pth.replace('.jpg', '.png')), center_i.astype(np.uint8))
                # plt.imsave(os.path.join(src_dir, 'test_l', 'source', 'scenes', pth.replace('B', 'L').replace('.jpg', '.png')), image2[0].astype(np.uint8))
                # plt.imsave(os.path.join(src_dir, 'test_r', 'source', 'scenes', pth.replace('B', 'R').replace('.jpg', '.png')), image2[1].astype(np.uint8))

                # percent = 10
                # delta = np.percentile(np.abs(flow_pr), percent)
                # mask = (flow_pr >= -delta) & (flow_pr <= delta)
                # masked_flow_pr = np.where(mask, 255, 0)

                # mask_rng_dir = os.path.join(masked_dir, f'{str(percent).replace('.','_')}')
                # os.makedirs(mask_rng_dir, exist_ok=True)
                # plt.imsave(os.path.join(mask_rng_dir, pth), masked_flow_pr.squeeze(), cmap='gray')
                

    eval_est.save_metrics()
    result = {**result, **eval_est.get_mean_metrics()}
    return result


@torch.no_grad()
def validate_QPD(model, datatype='dual', gt_types=['disp'], iters=32, mixed_prec=False, save_result=False, val_save_skip=1, image_set='test', path='', save_path='result/train', batch_size=1, preprocess_params={'crop_h':768, 'crop_w':960, 'resize_h': 224*4, 'resize_w':224*5}):
    """ Peform validation using the FlyingThings3D (TEST) split """
    model.eval()
    aug_params = {}
    
    if path == '':
        val_dataset = datasets.QPD(datatype=datatype, gt_types=gt_types, aug_params=aug_params, image_set=image_set, preprocess_params=preprocess_params)
    else:
        val_dataset = datasets.QPD(datatype=datatype, gt_types=gt_types, aug_params=aug_params, image_set=image_set, preprocess_params=preprocess_params, root=path)

    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, 
        pin_memory=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, drop_last=False)
    
    # val_loader = data.DataLoader(val_dataset, batch_size=batch_size, 
    #     pin_memory=True, num_workers=1, drop_last=False)
    
    log_dir = 'result'
    est_dir = os.path.join(log_dir, 'dp_est')
    gt_dir = os.path.join(log_dir, 'gt')
    os.makedirs(est_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    est_dir = os.path.join(save_path, 'est')
    err_dir = os.path.join(save_path, 'err')
    gt_dir = os.path.join(save_path, 'gt')
    src_dir = os.path.join(save_path, 'src')
    os.makedirs(est_dir, exist_ok=True)
    os.makedirs(err_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)


    path = os.path.basename(os.path.dirname(path))

    # ai2_bad_0_005px ~ ai2_bad_15px
    eval_est = Eval(os.path.join(save_path, 'center'), enabled_metrics=['epe', 'rmse', 'ai1', 'ai2', 'si', 'epe_bad_0_005px', 'epe_bad_0_01px', 'epe_bad_0_05px', 'epe_bad_0_1px', 'epe_bad_0_5px', 'epe_bad_1px'])
    
    result = {}

    if val_save_skip < batch_size:
        val_save_skip = 1
    else:
        val_save_skip = val_save_skip // batch_size

    # for val_id in tqdm(range(val_num)):
    for i_batch, data_blob in enumerate(tqdm(val_loader)):

        if i_batch % val_save_skip != 0:
            continue
        # if val_id == 2:
        #     break
        # paths, image1, image2, flow_gt, valid_gt = data_blob

        image_paths = data_blob['image_list']
        center = data_blob['center'].cuda()
        lrtb_list = data_blob['lrtb_list'].cuda()
        disp_gt =  data_blob['disp'].cuda()
        valid_gt = data_blob['disp_valid'].cuda()

        concat_lr = torch.cat([lrtb_list[:,0],lrtb_list[:,1]], dim=0).contiguous()
        
        with autocast(enabled=mixed_prec):
            _, flow_pr = model(center, concat_lr, iters=iters, test_mode=True)


        # Align dimensions and file format
        flow_pr = flow_pr.cpu().numpy()
        disp_gt = disp_gt.cpu().numpy()
        center = center.permute(0,2,3,1).cpu().numpy()
        
        # if flow_pr.shape[0]==2:
        #     flow_pr = flow_pr[1]-flow_pr[0]
        disp_gt = disp_gt/2

        assert flow_pr.shape == disp_gt.shape, (flow_pr.shape, disp_gt.shape)

        for i in range(batch_size):
            flow_pr_i = flow_pr[i]
            disp_gt_i = disp_gt[i]
            center_i = center[i]

            # fitting and save
            epe = eval_est.end_point_error(flow_pr_i, disp_gt_i)
            rmse = eval_est.root_mean_squared_error(flow_pr_i, disp_gt_i)
            # bads = eval_est.ai2_bad_pixel_metrics(flow_pr_i, disp_gt_i)
            bads = eval_est.epe_bad_pixel_metrics(flow_pr_i, disp_gt_i)
            est_ai1, est_b1 = eval_est.affine_invariant_1(flow_pr_i, disp_gt_i)
            est_ai2, est_b2 = eval_est.affine_invariant_2(flow_pr_i, disp_gt_i)
            si, alpha = eval_est.scale_invariant(flow_pr_i, disp_gt_i)
            est_ai2_fit = flow_pr_i * est_b2[0] + est_b2[1]

            val_id = i_batch * batch_size + i

            if save_result:
                if not os.path.exists('result/predictions/'+path+'/'):
                    os.makedirs('result/predictions/'+path+'/')
            
                pth_lists = image_paths[0][i].split('/')[-2:]
                pth = '/'.join(pth_lists)

                # Set range
                vmargin = 0.1
                vrng = disp_gt_i.max() - disp_gt_i.min()
                vmin, vmax = disp_gt_i.min() - vrng * vmargin, disp_gt_i.max() + vrng * vmargin
                vmin_err, vmax_err = 0, vrng * vmargin

                # vmin, vmax = np.min([est_ai2_fit, disp_gt_i]), np.max([est_ai2_fit, disp_gt_i])
                # vmin_err, vmax_err = np.min([est_ai2_fit - disp_gt_i, disp_gt_i - est_ai2_fit]), np.max([est_ai2_fit - disp_gt_i, disp_gt_i - est_ai2_fit])            
                eval_est.add_colorrange(vmin, vmax)

                os.makedirs(os.path.dirname(os.path.join(est_dir, pth)), exist_ok=True)
                os.makedirs(os.path.dirname(os.path.join(err_dir, pth)), exist_ok=True)
                os.makedirs(os.path.dirname(os.path.join(gt_dir, pth)), exist_ok=True)
                os.makedirs(os.path.dirname(os.path.join(src_dir, pth)), exist_ok=True)

                # Save in colormap
                plt.imsave(os.path.join(est_dir, pth), flow_pr_i.squeeze(), cmap='jet', vmin=vmin, vmax=vmax)        
                plt.imsave(os.path.join(err_dir, pth), np.abs(flow_pr_i.squeeze() - disp_gt_i.squeeze()), cmap='jet', vmin=vmin_err, vmax=vmax_err)
                
                plt.imsave(os.path.join(gt_dir, pth), disp_gt_i.squeeze(), cmap='jet', vmin=vmin, vmax=vmax)
                plt.imsave(os.path.join(src_dir, pth), center.astype(np.uint8))

                # Colorize된 이미지를 result에 저장
                colormap = cm.jet

                # est_ai2_fit colorized 이미지
                est_colorized = colormap((est_ai2_fit - vmin) / (vmax - vmin))  # 0~1 범위로 정규화
                est_colorized = (est_colorized * 255).astype(np.uint8)[0, :, :, :3]
                est_colorized = np.moveaxis(est_colorized, -1, 0)
                result[f'{val_id}_est_colormap'] = est_colorized

                # error 이미지 colorized
                error_image = np.abs(est_ai2_fit - disp_gt_i)
                error_colorized = colormap((error_image - vmin_err) / (vmax_err - vmin_err))
                error_colorized = (error_colorized * 255).astype(np.uint8)[0, :, :, :3]
                error_colorized = np.moveaxis(error_colorized, -1, 0)
                result[f'{val_id}_err_colormap'] = error_colorized

                # flow_gt_i colorized 이미지
                gt_colorized = colormap((disp_gt_i - vmin) / (vmax - vmin))
                gt_colorized = (gt_colorized * 255).astype(np.uint8)[0, :, :, :3]
                gt_colorized = np.moveaxis(gt_colorized, -1, 0)
                result[f'{val_id}_gt_colormap'] = gt_colorized


    eval_est.save_metrics()
    result = {**result, **eval_est.get_mean_metrics()}

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--dataset', help="dataset for evaluation", required=False, choices=["QPD-Test", "QPD-Valid", "Real_QPD", "DPD_Disp"], default="QPD")
    parser.add_argument('--datasets_path', default='/mnt/d/Mono+Dual/QP-Data', help="test datasets.")    
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=8, help='number of flow-field updates during forward pass')
    parser.add_argument('--input_image_num', type=int, default=2, help="2 for LR and 4 for LRTB")
    parser.add_argument('--CAPA', default=True, help="if use Channel wise and pixel wise attention")

    # Architecure choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--save_result', type=bool, default='False')
    parser.add_argument('--save_name', default='val')
    parser.add_argument('--save_path', default='result/validations/eval.txt')

    # Data settings
    parser.add_argument('--qpd_valid_bs', type=int, default=1)
    parser.add_argument('--qpd_test_bs', type=int, default=1)
    parser.add_argument('--real_qpd_bs', type=int, default=1)
    parser.add_argument('--dp_disp_bs', type=int, default=1)
    parser.add_argument('--datatype', type=str, default='dual', help='dual or quad')

    # Depth Anything V2
    parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--feature_converter', type=str, default='', help="training datasets.")

    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    
    args = parser.parse_args()

    # args.save_result = args.save_result == str(True)

    # # Argument categorization
    # da_v2_keys = {'encoder', 'img-size', 'epochs', 'local-rank', 'port', 'restore_ckpt_da_v2', 'freeze_da_v2'}
    # else_keys = {'name', 'restore_ckpt_da_v2', 'restore_ckpt_qpd_net', 'mixed_precision', 'batch_size', 'train_datasets', 'datasets_path', 'lr', 'num_steps', 'input_image_num', 'image_size', 'train_iters', 'wdecay', 'CAPA', 'valid_iters', 'corr_implementation', 'shared_backbone', 'corr_levels', 'corr_radius', 'n_downsample', 'context_norm', 'slow_fast_gru', 'n_gru_layers', 'hidden_dims', 'img_gamma', 'saturation_range', 'do_flip', 'spatial_scale', 'noyjitter', 'feature_converter', 'save_path'}

    # def split_arguments(args):
    #     args_dict = vars(args)
    #     da_v2_args = {key: args_dict[key] for key in da_v2_keys if key in args_dict}
    #     else_args = {key: args_dict[key] for key in args_dict if key in else_keys}

    #     return {
    #         'da_v2': Namespace(**da_v2_args),
    #         'else': Namespace(**else_args),
    #     }
    
    # split_args = split_arguments(args)
    
    model = MonoQPD(args)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')


    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        # model.load_state_dict(checkpoint, strict=True)
        if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint and 'scheduler_state_dict' in checkpoint:
            c={}
            c['model_state_dict'] = fix_key(checkpoint['model_state_dict'])
            model.load_state_dict(c['model_state_dict'])
        else:
            model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    # Delete after mdoel is properly saved
    model = nn.DataParallel(model)

    # # Delete after mdoel is properly saved
    # model = model.module


    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")

    use_mixed_precision = args.corr_implementation.endswith("_cuda")


    if 'QPD-Test' == args.dataset:
        save_path = os.path.join(args.save_path, 'qpd-test', os.path.basename(args.restore_ckpt).replace('.pth', ''))
        print(save_path)
        result = validate_QPD(model, iters=args.valid_iters, mixed_prec=use_mixed_precision, save_result=args.save_result, datatype = args.datatype, image_set="test", path='datasets/QP-Data', save_path=save_path, batch_size=args.qpd_test_bs if args.qpd_test_bs else 1)
    if 'QPD-Valid' == args.dataset:
        save_path = os.path.join(args.save_path, 'qpd-valid', os.path.basename(args.restore_ckpt).replace('.pth', ''))
        print(save_path)
        result = validate_QPD(model, iters=args.valid_iters, mixed_prec=use_mixed_precision, save_result=args.save_result, datatype = args.datatype, image_set="validation", path='datasets/QP-Data', save_path=save_path, batch_size=args.qpd_valid_bs if args.qpd_valid_bs else 1)
    if 'DPD_Disp' == args.dataset:
        save_path = os.path.join(args.save_path, 'dp-disp', os.path.basename(args.restore_ckpt).replace('.pth', ''))
        print(save_path)
        result = validate_DPD_Disp(model, iters=args.valid_iters, mixed_prec=use_mixed_precision, save_result=args.save_result, datatype = args.datatype, image_set="test", path='datasets/MDD_dataset', save_path=save_path, batch_size=args.dp_disp_bs if args.dp_disp_bs else 1)
    if 'Real_QPD' == args.dataset:
        save_path = os.path.join(args.save_path, 'real-qpd-test', os.path.basename(args.restore_ckpt).replace('.pth', ''))
        print(save_path)
        result = validate_Real_QPD(model, iters=args.valid_iters, mixed_prec=use_mixed_precision, save_result=args.save_result, datatype = args.datatype, image_set="test", path='datasets/Real-QP-Data', save_path=save_path, batch_size=args.real_qpd_bs if args.real_qpd_bs else 1)

    print(result)

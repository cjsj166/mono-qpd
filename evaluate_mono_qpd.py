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

from exp_args_settings.utils import get_ckpts_in_dir
from exp_args_settings.train_settings import get_train_config

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

        current_batch_size = flow_pr.shape[0]
        for i in range(current_batch_size):
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

    ai2_fit_dir = os.path.join(save_path, 'ai2_fit')
    ai2_dir = os.path.join(save_path, 'ai2')
    gt_dir = os.path.join(save_path, 'gt')
    src_dir = os.path.join(save_path, 'src')
    src_test_c_dir = os.path.join(src_dir, 'test_c', 'source', 'scenes')
    os.makedirs(ai2_fit_dir, exist_ok=True)
    os.makedirs(ai2_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(src_test_c_dir, exist_ok=True)

    eval_est = Eval(os.path.join(save_path, 'center'), enabled_metrics=['ai1', 'ai2', 'sc', 'ai2_bad_0_003', 'ai2_bad_0_005', 'ai2_bad_0_01', 'ai2_bad_0_03', 'ai2_bad_0_05'])

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

        current_batch_size = flow_pr.shape[0]
        for i in range(current_batch_size):
            flow_pr_i = flow_pr[i]
            inv_depth_gt_i = inv_depth_gt[i]
            center_i = center[i]
            est_ai1, est_b1 = eval_est.affine_invariant_1(flow_pr_i, inv_depth_gt_i)
            est_ai2, est_b2 = eval_est.affine_invariant_2(flow_pr_i, inv_depth_gt_i)
            sc = eval_est.spearman_correlation(flow_pr_i, inv_depth_gt_i)
            bads = eval_est.ai2_bad_pixel_metrics(flow_pr_i, inv_depth_gt_i)
            est_ai2_fit = flow_pr_i * est_b2[0] + est_b2[1]

            val_id = i_batch * batch_size + i

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
                err_rng = 0.7
                vmin_err, vmax_err = 0, vrng * err_rng

                eval_est.add_colorrange(vmin, vmax)

                
                os.makedirs('result/MVA_submission', exist_ok=True)

                with open('result/MVA_submission/dpd_disp_affine_fit_range.txt', 'a') as f:
                    f.write(f'{val_id}: {vmin}, {vmax}\n')

                with open('result/MVA_submission/dpd_disp_ai2_range.txt', 'a') as f:
                    f.write(f'{val_id}: {vmin_err}, {vmax_err}\n')


                # Save in colormap
                plt.imsave(os.path.join(ai2_fit_dir, pth), est_ai2_fit.squeeze(), cmap='jet', vmin=vmin, vmax=vmax)
                plt.imsave(os.path.join(ai2_dir, pth), np.abs(est_ai2_fit.squeeze() - inv_depth_gt_i.squeeze()), cmap='jet', vmin=vmin_err, vmax=vmax_err)
                
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
def validate_QPD(model, datatype='dual', gt_types=['disp'], iters=32, mixed_prec=False, save_result=False, val_save_skip=1, image_set='test', path='', save_path='result/train', batch_size=1, preprocess_params={'crop_h':672, 'crop_w':896, 'resize_h': 672, 'resize_w':896}):
    """ Peform validation using the FlyingThings3D (TEST) split """
    model.eval()
    aug_params = {}
    
    if path == '':
        val_dataset = datasets.QPD(datatype=datatype, gt_types=gt_types, aug_params=aug_params, image_set=image_set, preprocess_params=preprocess_params)
    else:
        val_dataset = datasets.QPD(datatype=datatype, gt_types=gt_types, aug_params=aug_params, image_set=image_set, preprocess_params=preprocess_params, root=path)

    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, 
        pin_memory=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, drop_last=False)
    
    disp_dir = os.path.join(save_path, 'disp')
    epe_dir = os.path.join(save_path, 'epe')
    epe0_3_dir = os.path.join(save_path, 'epe0_3')
    epe0_5_dir = os.path.join(save_path, 'epe0_5')
    ai2_fit_dir = os.path.join(save_path, 'ai2_fit')
    ai2_dir = os.path.join(save_path, 'ai2')
    ai2_0_3_dir = os.path.join(save_path, 'ai2_0_3')
    ai2_0_5_dir = os.path.join(save_path, 'ai2_0_5')
    gt_dir = os.path.join(save_path, 'gt')
    src_dir = os.path.join(save_path, 'src')
    os.makedirs(epe_dir, exist_ok=True)
    os.makedirs(epe0_3_dir, exist_ok=True)
    os.makedirs(epe0_5_dir, exist_ok=True)
    os.makedirs(ai2_0_3_dir, exist_ok=True)
    os.makedirs(ai2_0_5_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)

    path = os.path.basename(os.path.dirname(path))

    eval_est = Eval(os.path.join(save_path, 'center'), enabled_metrics=['epe', 'rmse', 'ai1', 'ai2', 'si', 'epe_bad_0_005', 'epe_bad_0_01', 'epe_bad_0_05', 'epe_bad_0_1', 'epe_bad_0_5', 'epe_bad_1'])
    
    result = {}

    if val_save_skip < batch_size:
        val_save_skip = 1
    else:
        val_save_skip = val_save_skip // batch_size

    for i_batch, data_blob in enumerate(tqdm(val_loader)):

        if i_batch % val_save_skip != 0:
            continue

        image_paths = data_blob['image_list']
        center = data_blob['center'].cuda()
        lrtb_list = data_blob['lrtb_list'].cuda()
        disp_gt =  data_blob['disp'].cuda()
        valid_gt = data_blob['disp_valid'].cuda()

        concat_lr = torch.cat([lrtb_list[:,0],lrtb_list[:,1]], dim=0).contiguous()
        
        with autocast(enabled=mixed_prec):
            _, flow_pr = model(center, concat_lr, iters=iters, test_mode=True)

        flow_pr = flow_pr.cpu().numpy()
        disp_gt = disp_gt.cpu().numpy()
        center = center.permute(0,2,3,1).cpu().numpy()
        
        disp_gt = disp_gt / 2

        assert flow_pr.shape == disp_gt.shape, (flow_pr.shape, disp_gt.shape)

        current_batch_size = flow_pr.shape[0]
        for i in range(current_batch_size):
            flow_pr_i = flow_pr[i]
            disp_gt_i = disp_gt[i]
            center_i = center[i]

            epe = eval_est.end_point_error(flow_pr_i, disp_gt_i)
            rmse = eval_est.root_mean_squared_error(flow_pr_i, disp_gt_i)
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

                os.makedirs(os.path.dirname(os.path.join(disp_dir, pth)), exist_ok=True)
                os.makedirs(os.path.dirname(os.path.join(ai2_dir, pth)), exist_ok=True)
                os.makedirs(os.path.dirname(os.path.join(ai2_fit_dir, pth)), exist_ok=True)
                os.makedirs(os.path.dirname(os.path.join(epe_dir, pth)), exist_ok=True)
                os.makedirs(os.path.dirname(os.path.join(epe0_3_dir, pth)), exist_ok=True)
                os.makedirs(os.path.dirname(os.path.join(epe0_5_dir, pth)), exist_ok=True)
                os.makedirs(os.path.dirname(os.path.join(ai2_0_3_dir, pth)), exist_ok=True)
                os.makedirs(os.path.dirname(os.path.join(ai2_0_5_dir, pth)), exist_ok=True)
                os.makedirs(os.path.dirname(os.path.join(gt_dir, pth)), exist_ok=True)
                os.makedirs(os.path.dirname(os.path.join(src_dir, pth)), exist_ok=True)

                vrng = disp_gt_i.max() - disp_gt_i.min()

                vmargin = 0.1
                vmin, vmax = disp_gt_i.min() - vrng * vmargin, disp_gt_i.max() + vrng * vmargin
                eval_est.add_colorrange(vmin, vmax)
                plt.imsave(os.path.join(disp_dir, pth), flow_pr_i.squeeze(), cmap='jet', vmin=vmin, vmax=vmax)
                plt.imsave(os.path.join(ai2_fit_dir, pth), est_ai2_fit.squeeze(), cmap='jet', vmin=vmin, vmax=vmax)

                os.makedirs('result/MVA_submission', exist_ok=True)

                with open('result/MVA_submission/qpd-test_affine_fit_range.txt', 'a') as f:
                    f.write(f'{val_id}: {vmin}, {vmax}\n')

                err_rat = 0.7
                vmin_err, vmax_err = 0, vrng * err_rat
                plt.imsave(os.path.join(epe_dir, pth), np.abs(flow_pr_i.squeeze() - disp_gt_i.squeeze()), cmap='jet', vmin=vmin_err, vmax=vmax_err)
                plt.imsave(os.path.join(ai2_dir, pth), np.abs(est_ai2_fit.squeeze() - disp_gt_i.squeeze()), cmap='jet', vmin=vmin_err, vmax=vmax_err)

                with open('result/MVA_submission/qpd-test_ai2_range.txt', 'a') as f:
                    f.write(f'{val_id}: {vmin_err}, {vmax_err}\n')

                err_rat_0_3 = 0.3
                vmin_err_0_3, vmax_err_0_3 = 0, vrng * err_rat_0_3
                plt.imsave(os.path.join(epe0_3_dir, pth), np.abs(flow_pr_i.squeeze() - disp_gt_i.squeeze()), cmap='jet', vmin=vmin_err_0_3, vmax=vmax_err_0_3)
                plt.imsave(os.path.join(ai2_0_3_dir, pth), np.abs(est_ai2_fit.squeeze() - disp_gt_i.squeeze()), cmap='jet', vmin=vmin_err_0_3, vmax=vmax_err_0_3)

                err_rat_0_5 = 0.5
                vmin_err_0_5, vmax_err_0_5 = 0, vrng * err_rat_0_5
                plt.imsave(os.path.join(epe0_5_dir, pth), np.abs(flow_pr_i.squeeze() - disp_gt_i.squeeze()), cmap='jet', vmin=vmin_err_0_5, vmax=vmax_err_0_5)
                plt.imsave(os.path.join(ai2_0_5_dir, pth), np.abs(est_ai2_fit.squeeze() - disp_gt_i.squeeze()), cmap='jet', vmin=vmin_err_0_5, vmax=vmax_err_0_5)

                plt.imsave(os.path.join(gt_dir, pth), disp_gt_i.squeeze(), cmap='jet', vmin=vmin, vmax=vmax)
                plt.imsave(os.path.join(src_dir, pth), center_i.astype(np.uint8))

                colormap = cm.jet

                est_colorized = colormap((est_ai2_fit - vmin) / (vmax - vmin))
                est_colorized = (est_colorized * 255).astype(np.uint8)[0, :, :, :3]
                est_colorized = np.moveaxis(est_colorized, -1, 0)
                result[f'{val_id}_est_colormap'] = est_colorized

                error_image = np.abs(est_ai2_fit - disp_gt_i)
                error_colorized = colormap((error_image - vmin_err) / (vmax_err - vmin_err))
                error_colorized = (error_colorized * 255).astype(np.uint8)[0, :, :, :3]
                error_colorized = np.moveaxis(error_colorized, -1, 0)
                result[f'{val_id}_err_colormap'] = error_colorized

                gt_colorized = colormap((disp_gt_i - vmin) / (vmax - vmin))
                gt_colorized = (gt_colorized * 255).astype(np.uint8)[0, :, :, :3]
                gt_colorized = np.moveaxis(gt_colorized, -1, 0)
                result[f'{val_id}_gt_colormap'] = gt_colorized

    eval_est.save_metrics()
    result = {**result, **eval_est.get_mean_metrics()}

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='Interp', help="name your experiment")
    parser.add_argument('--ckpt_epoch', type=int, default=0)
    parser.add_argument('--eval_datasets', choices=['QPD-Test', 'QPD-Valid', 'DPD_Disp', 'Real_QPD', 'QPD-Test-noise'], nargs='+', default=[], required=True, help="Additional dataset to evaluate")
    
    args = parser.parse_args()

    conf = get_train_config(args.exp_name)

    ckpts = get_ckpts_in_dir(conf.save_path)

    for ckpt in ckpts:
        try:
            epoch = int(os.path.basename(ckpt).split('_')[0])
        except Exception as e:
            print(f'{e} occured from ckpt: {ckpt}')

        if epoch == args.ckpt_epoch:
            restore_ckpt = ckpt
            break    

    model = MonoQPD(conf)
    if restore_ckpt is not None:
        assert str(restore_ckpt).endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(restore_ckpt)
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

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")
    use_mixed_precision = conf.corr_implementation.endswith("_cuda")

    if 'QPD-Test' in args.eval_datasets:
        save_path = os.path.join(conf.save_path, 'qpd-test', str(restore_ckpt.name).replace('.pth', ''))
        print(save_path)
        result = validate_QPD(model, iters=conf.valid_iters, mixed_prec=use_mixed_precision, save_result=True, datatype = conf.datatype, image_set="test", path='datasets/QP-Data', save_path=save_path, batch_size=conf.qpd_test_bs if conf.qpd_test_bs else 1)

    if 'QPD-Test-noise' in args.eval_datasets:
        save_path = os.path.join(conf.save_path, 'qpd-test-noise', str(restore_ckpt.name).replace('.pth', ''))
        print(save_path)
        result = validate_QPD(model, iters=conf.valid_iters, mixed_prec=use_mixed_precision, save_result=True, datatype = conf.datatype, image_set="test", path='datasets/QP-Data-noise0.001', save_path=save_path, batch_size=conf.qpd_test_bs if conf.qpd_test_bs else 1)

    if 'QPD-Valid' in args.eval_datasets:
        save_path = os.path.join(conf.save_path, 'qpd-valid', str(restore_ckpt.name).replace('.pth', ''))
        print(save_path)
        result = validate_QPD(model, iters=conf.valid_iters, mixed_prec=use_mixed_precision, save_result=True, datatype = conf.datatype, image_set="validation", path='datasets/QP-Data', save_path=save_path, batch_size=conf.qpd_valid_bs if conf.qpd_valid_bs else 1)
    if 'DPD_Disp' in args.eval_datasets:
        save_path = os.path.join(conf.save_path, 'dp-disp', str(restore_ckpt.name).replace('.pth', ''))
        print(save_path)
        result = validate_DPD_Disp(model, iters=conf.valid_iters, mixed_prec=use_mixed_precision, save_result=True, datatype = conf.datatype, image_set="test", path='datasets/MDD_dataset', save_path=save_path, batch_size=conf.dp_disp_bs if conf.dp_disp_bs else 1)
    if 'Real_QPD' in args.eval_datasets:
        save_path = os.path.join(conf.save_path, 'real-qpd-test', str(restore_ckpt.name).replace('.pth', ''))
        print(save_path)
        result = validate_Real_QPD(model, iters=conf.valid_iters, mixed_prec=use_mixed_precision, save_result=True, datatype = conf.datatype, image_set="test", path='datasets/Real-QP-Data', save_path=save_path, batch_size=conf.real_qpd_bs if conf.real_qpd_bs else 1)

    print(result)

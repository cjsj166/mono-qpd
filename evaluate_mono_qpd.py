from __future__ import print_function, division
import sys
sys.path.append('core')

import torch.optim as optim
import argparse
import time
import logging
import numpy as np
import glob
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
from copy import deepcopy

from metrics.eval import Eval
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

from exp_args_settings.utils import get_ckpts_in_dir

from runsync.presets import get_run_setting

class EvalLogger:
    def __init__(self, log_dir='result/runs', epoch: int = 0):
        """
        Evaluation 전용 Logger
        - epoch: 기록 시점의 epoch
        """
        self.epoch = epoch
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir))

    def write_dict(self, results: dict):
        """
        딕셔너리 형태의 평가 결과 기록
        - 스칼라는 scalar로
        - 이미지/배열은 image로
        """
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join('result/runs'))

        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                if value.dim() == 4:
                    value = value[0]
                self.writer.add_image(key, value, global_step=self.epoch)
            elif isinstance(value, np.ndarray):
                if value.ndim == 4:
                    value = value[0]
                self.writer.add_image(key, value, global_step=self.epoch)
            else:
                self.writer.add_scalar(key, value, global_step=self.epoch)

    def close(self):
        if self.writer:
            self.writer.flush()
            self.writer.close()





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

        # if i_batch > 3:
        #     break

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
        center = center[..., 32:h-32, 32:w-32]

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
            
            pth_lists = image_paths[0][i].split('/')
            pth = '/'.join(pth_lists[-2:])
            filename = os.path.join(save_path.replace('result/train/', ''), pth) 
            eval_est.add_filename(filename)

            # print(est_ai1, est_b1, est_ai2, est_b2, sc)

            val_id = i_batch * batch_size + i

            # print(flow_pr_i[0][418:421][317:343])

            # Set range
            vmargin = 0.3
            vrng = inv_depth_gt_i.max() - inv_depth_gt_i.min()
            vmin, vmax = inv_depth_gt_i.min() - vrng * vmargin, inv_depth_gt_i.max() + vrng * vmargin
            vmin = 0 if vmin < 0 else vmin
            err_rng = 0.7
            vmin_err, vmax_err = 0, vrng * err_rng

            eval_est.add_colorrange(vmin, vmax)

            if save_result:
                if not os.path.exists('result/predictions/'+path+'/'):
                    os.makedirs('result/predictions/'+path+'/')
                
                pth_lists = image_paths[0][i].split('/')[-3:]
                pth = '/'.join(pth_lists)
                pth = os.path.basename(pth)


                # Save in colormap
                plt.imsave(os.path.join(ai2_fit_dir, pth), est_ai2_fit.squeeze(), cmap='jet_r', vmin=vmin, vmax=vmax)
                plt.imsave(os.path.join(ai2_dir, pth), np.abs(est_ai2_fit.squeeze() - inv_depth_gt_i.squeeze()), cmap='jet', vmin=vmin_err, vmax=vmax_err)
                
                plt.imsave(os.path.join(gt_dir, pth), inv_depth_gt_i.squeeze(), cmap='jet_r', vmin=vmin, vmax=vmax)

                plt.imsave(os.path.join(src_test_c_dir, pth.replace('.jpg', '.png')), center_i.astype(np.uint8))

                # img_est_ai2_fit = Image.open(os.path.join(ai2_fit_dir, pth)).convert("RGB")
                # img_est_ai2_fit = np.array(img_est_ai2_fit)
                # img_est_ai2_fit = np.moveaxis(img_est_ai2_fit, -1, 0)
                # result[f'img/{val_id}/est_ai2_fit'] = img_est_ai2_fit
                # img_gt = Image.open(os.path.join(gt_dir, pth)).convert("RGB")
                # img_gt = np.array(img_gt)
                # img_gt = np.moveaxis(img_gt, -1, 0)
                # result[f'img/{val_id}/gt'] = img_gt
                # img_src = Image.open(os.path.join(src_test_c_dir, pth.replace('.jpg', '.png'))).convert("RGB")
                # img_src = np.array(img_src)
                # img_src = np.moveaxis(img_src, -1, 0)
                # result[f'img/{val_id}/src'] = img_src
                

    eval_est.save_metrics()
    result = {**result, **eval_est.get_mean_metrics()}
    return result

@torch.no_grad()
def validate_DP119(model, datatype='dual', gt_types=['inv_depth'], iters=32, mixed_prec=False, save_result=False, val_save_skip=1, image_set='test', path='', save_path='result/predictions', batch_size=1, preprocess_params={'crop_h':3000, 'crop_w':4000, 'resize_h': 224*3, 'resize_w':224*4}):
    model.eval()
    aug_params = {}
    
    if path == '':
        val_dataset = datasets.DP119(datatype=datatype, gt_types=gt_types, aug_params=aug_params, preprocess_params=preprocess_params, image_set=image_set)
    else:
        val_dataset = datasets.DP119(datatype=datatype, gt_types=gt_types, aug_params=aug_params, image_set=image_set, preprocess_params=preprocess_params, root=path)

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

        # if i_batch > 3:
        #     break

        image_paths = data_blob['image_list']
        center = data_blob['center'].cuda()
        lrtb_list = data_blob['lrtb_list'].cuda()
        inv_depth_gt =  data_blob['inv_depth'].cuda()
        valid_gt = data_blob['inv_depth_valid'].cuda()

        concat_lr = torch.cat([lrtb_list[:,0],lrtb_list[:,1]], dim=0).contiguous()

        with autocast(enabled=mixed_prec):
            _, flow_pr = model(center, concat_lr, iters=iters, test_mode=True)

        # Crop invalid regions
        # h, w = flow_pr.shape[-2:]
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
            
            pth_lists = image_paths[0][i].split('/')
            pth = '/'.join(pth_lists[-2:])
            filename = os.path.join(save_path.replace('result/train/', ''), pth) 
            eval_est.add_filename(filename)

            # print(est_ai1, est_b1, est_ai2, est_b2, sc)

            val_id = i_batch * batch_size + i

            # print(flow_pr_i[0][418:421][317:343])

            # Set range
            vmargin = 0.3
            vrng = inv_depth_gt_i.max() - inv_depth_gt_i.min()
            vmin, vmax = inv_depth_gt_i.min() - vrng * vmargin, inv_depth_gt_i.max() + vrng * vmargin
            vmin = 0 if vmin < 0 else vmin
            err_rng = 0.7
            vmin_err, vmax_err = 0, vrng * err_rng

            eval_est.add_colorrange(vmin, vmax)

            if save_result:
                if not os.path.exists('result/predictions/'+path+'/'):
                    os.makedirs('result/predictions/'+path+'/')
                
                pth_lists = image_paths[0][i].split('/')[-3:]
                pth = '/'.join(pth_lists)
                pth = os.path.basename(pth)


                # Save in colormap
                plt.imsave(os.path.join(ai2_fit_dir, pth), est_ai2_fit.squeeze(), cmap='jet_r', vmin=vmin, vmax=vmax)
                plt.imsave(os.path.join(ai2_dir, pth), np.abs(est_ai2_fit.squeeze() - inv_depth_gt_i.squeeze()), cmap='jet', vmin=vmin_err, vmax=vmax_err)
                
                plt.imsave(os.path.join(gt_dir, pth), inv_depth_gt_i.squeeze(), cmap='jet_r', vmin=vmin, vmax=vmax)

                plt.imsave(os.path.join(src_test_c_dir, pth.replace('.jpg', '.png')), center_i.astype(np.uint8))


    eval_est.save_metrics()
    result = {**result, **eval_est.get_mean_metrics()}
    return result


@torch.no_grad()
def validate_DP5K(model, datatype='dual', gt_types=['disp'], iters=32, mixed_prec=False, save_result=False, val_save_skip=1, image_set='test', path='', save_path='result/predictions', batch_size=1, preprocess_params={'crop_h':1120, 'crop_w':1568, 'resize_h':1120, 'resize_w':1568}):
    """ Peform validation using the FlyingThings3D (TEST) split """
    model.eval()
    aug_params = {}
    
    if path == '':
        val_dataset = datasets.DP5K(datatype=datatype, gt_types=gt_types, aug_params=aug_params, preprocess_params=preprocess_params, image_set=image_set)
    else:
        val_dataset = datasets.DP5K(datatype=datatype, gt_types=gt_types, aug_params=aug_params, image_set=image_set, preprocess_params=preprocess_params, root=path)

    # val_loader = data.DataLoader(val_dataset, batch_size=batch_size, 
    #     pin_memory=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, drop_last=False)
    
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, 
        pin_memory=True, num_workers=0, drop_last=False)    


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
        # if i_batch > 3:
        #     break

        image_paths = data_blob['image_list']
        center = data_blob['center'].cuda()
        lrtb_list = data_blob['lrtb_list'].cuda()
        depth_gt =  -data_blob['disp'].cuda()
        valid_gt = data_blob['disp_valid'].cuda()

        concat_lr = torch.cat([lrtb_list[:,0],lrtb_list[:,1]], dim=0).contiguous()
        
        with autocast(enabled=mixed_prec):
            _, flow_pr = model(center, concat_lr, iters=iters, test_mode=True)

        # Crop invalid regions
        h, w = flow_pr.shape[-2:]
        # flow_pr = flow_pr[..., 32:h-32, 32:w-32]
        # depth_gt = depth_gt[..., 32:h-32, 32:w-32]
        # center = center[..., 32:h-32, 32:w-32]

        # flow_pr = torch.zeros_like(flow_gt)

        # Align dimensions and file format
        flow_pr = flow_pr.cpu().numpy()
        depth_gt = depth_gt.cpu().numpy()
        center = center.permute(0,2,3,1).cpu().numpy()
        
        assert flow_pr.shape == depth_gt.shape, (flow_pr.shape, depth_gt.shape)

        current_batch_size = flow_pr.shape[0]
        for i in range(current_batch_size):
            flow_pr_i = flow_pr[i]
            depth_gt_i = depth_gt[i]
            depth_conf = depth_gt_i > 0

            depth_gt_i /= 100.0 # convert to meters
            inv_depth_gt_i = np.zeros_like(depth_gt_i)
            inv_depth_gt_i[depth_conf] = 1 / depth_gt_i[depth_conf] # convert to 1/meters

            center_i = center[i]
            est_ai1, est_b1 = eval_est.affine_invariant_1(flow_pr_i, inv_depth_gt_i, confidence_map=depth_conf)
            est_ai2, est_b2 = eval_est.affine_invariant_2(flow_pr_i, inv_depth_gt_i, confidence_map=depth_conf)
            sc = eval_est.spearman_correlation(flow_pr_i, inv_depth_gt_i, confidence_map=depth_conf)
            bads = eval_est.ai2_bad_pixel_metrics(flow_pr_i, inv_depth_gt_i, confidence_map=depth_conf)
            est_ai2_fit = flow_pr_i * est_b2[0] + est_b2[1]
            
            pth_lists = image_paths[0][i].split('/')
            pth = '/'.join(pth_lists[-2:])
            filename = os.path.join(save_path.replace('result/train/', ''), pth) 
            eval_est.add_filename(filename)

            val_id = i_batch * batch_size + i

            # Set range
            vmargin = 0.3
            vrng = inv_depth_gt_i[depth_conf].max() - inv_depth_gt_i[depth_conf].min()
            vmin, vmax = inv_depth_gt_i[depth_conf].min() - vrng * vmargin, inv_depth_gt_i[depth_conf].max() + vrng * vmargin
            vmin = 0 if vmin < 0 else vmin
            
            err_margin = 0.3
            vmin_err, vmax_err = 0, vrng * err_margin
            eval_est.add_colorrange(vmin, vmax)

            if save_result:
                if not os.path.exists('result/predictions/'+path+'/'):
                    os.makedirs('result/predictions/'+path+'/')
                
                pth_lists = image_paths[0][i].split('/')[-3:]
                pth = '/'.join(pth_lists)
                


                # Save in colormap
                os.makedirs(os.path.join(ai2_fit_dir, os.path.dirname(pth)), exist_ok=True)
                plt.imsave(os.path.join(ai2_fit_dir, pth), est_ai2_fit.squeeze(), cmap='jet', vmin=vmin, vmax=vmax)

                ai2_err = np.ones_like(inv_depth_gt_i) * -100
                ai2_err[depth_conf] = np.abs(est_ai2_fit[depth_conf] - inv_depth_gt_i[depth_conf])
                os.makedirs(os.path.join(ai2_dir, os.path.dirname(pth)), exist_ok=True)
                plt.imsave(os.path.join(ai2_dir, pth), ai2_err.squeeze(), cmap='jet', vmin=vmin_err, vmax=vmax_err)
                ai2_err_color = np.array(Image.open(os.path.join(ai2_dir, pth)))
                ai2_err_color[~depth_conf.squeeze()] = [0, 0, 0, 255]
                plt.imsave(os.path.join(ai2_dir, pth), ai2_err_color)

                os.makedirs(os.path.join(gt_dir, os.path.dirname(pth)), exist_ok=True)
                plt.imsave(os.path.join(gt_dir, pth), inv_depth_gt_i.squeeze(), cmap='jet', vmin=vmin, vmax=vmax)
                gt_color = np.array(Image.open(os.path.join(gt_dir, pth)))
                gt_color[~depth_conf.squeeze()] = [0, 0, 0, 255]
                plt.imsave(os.path.join(gt_dir, pth), gt_color)

                os.makedirs(os.path.join(src_test_c_dir, os.path.dirname(pth)), exist_ok=True)
                plt.imsave(os.path.join(src_test_c_dir, pth.replace('.jpg', '.png')), center_i.astype(np.uint8))

    eval_est.save_metrics()
    result = {**result, **eval_est.get_mean_metrics()}
    return result

@torch.no_grad()
def validate_QPD_FStops(model, datatype='dual', gt_types=['disp'], iters=32, mixed_prec=False, save_result=False, val_save_skip=1, image_set='test', path='datasets/qpd-test-fstops', save_path='result/train', batch_size=1, preprocess_params={'crop_h':672, 'crop_w':896, 'resize_h': 672, 'resize_w':896}):
    """ Perform validation using multiple f-stop datasets """
    model.eval()
    
    # Find all f-stop directories
    fstop_dirs = glob.glob(os.path.join(path, 'qpd-test-fstop-*'))
    if not fstop_dirs:
        raise ValueError(f"No f-stop directories found in {path}")
    
    fstop_dirs.sort()  # Ensure consistent ordering
    print(f"Found {len(fstop_dirs)} f-stop directories")
    
    # Dictionary to store results for each f-stop and overall average
    all_results = {}
    fstop_metrics = {}
    
    # Process each f-stop directory
    for fstop_dir in tqdm(fstop_dirs, desc="Processing f-stops"):
        # Extract f-stop value from directory name (e.g., qpd-test-fstop-0_1 -> 0_1)
        fstop_value = os.path.basename(fstop_dir).replace('qpd-test-fstop-', '')
        print(f"\nProcessing f-stop {fstop_value}: {fstop_dir}")
        
        # Create dataset for this f-stop
        aug_params = {}
        val_dataset = datasets.QPD(
            datatype=datatype, 
            gt_types=gt_types, 
            aug_params=aug_params, 
            image_set=image_set, 
            preprocess_params=preprocess_params, 
            root=fstop_dir
        )
        
        val_loader = data.DataLoader(
            val_dataset, 
            batch_size=batch_size,
            pin_memory=True, 
            num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, 
            drop_last=False
        )
        
        # Create save directories for this f-stop
        fstop_save_path = save_path
        disp_dir = os.path.join(fstop_save_path, 'disp')
        epe_dir = os.path.join(fstop_save_path, 'epe')
        epe0_3_dir = os.path.join(fstop_save_path, 'epe0_3')
        epe0_5_dir = os.path.join(fstop_save_path, 'epe0_5')
        ai2_fit_dir = os.path.join(fstop_save_path, 'ai2_fit')
        ai2_dir = os.path.join(fstop_save_path, 'ai2')
        ai2_0_3_dir = os.path.join(fstop_save_path, 'ai2_0_3')
        ai2_0_5_dir = os.path.join(fstop_save_path, 'ai2_0_5')
        gt_dir = os.path.join(fstop_save_path, 'gt')
        src_dir = os.path.join(fstop_save_path, 'src')
        
        # Create eval object for this f-stop
        eval_est = Eval(
            os.path.join(fstop_save_path, 'center'), 
            enabled_metrics=['epe', 'rmse', 'ai1', 'ai2', 'si', 'epe_bad_0_005', 'epe_bad_0_01', 'epe_bad_0_05', 'epe_bad_0_1', 'epe_bad_0_5', 'epe_bad_1']
        )
        
        if val_save_skip < batch_size:
            val_save_skip = 1
        else:
            val_save_skip = val_save_skip // batch_size
        
        # Quantile edges for binned EPE evaluation
        quantile_edges = np.array([-1.5, -1.125, -0.75, -0.5625, -0.375, -0.28125, -0.1875, 0.005859, 0.251953, 0.333984, 0.486328, 0.597656, 0.75, 0.84375, 0.9375, 1.125])
        eval_est.bin_edges = quantile_edges
        
        # Process batches for this f-stop
        for i_batch, data_blob in enumerate(val_loader):
            if i_batch % val_save_skip != 0:
                continue
                
            image_paths = data_blob['image_list']
            center = data_blob['center'].cuda()
            lrtb_list = data_blob['lrtb_list'].cuda()
            disp_gt = data_blob['disp'].cuda()
            valid_gt = data_blob['disp_valid'].cuda()

            concat_lr = torch.cat([lrtb_list[:,0], lrtb_list[:,1]], dim=0).contiguous()
            
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

                # Calculate metrics
                epe = eval_est.end_point_error(flow_pr_i, disp_gt_i)
                rmse = eval_est.root_mean_squared_error(flow_pr_i, disp_gt_i)
                bads = eval_est.epe_bad_pixel_metrics(flow_pr_i, disp_gt_i)
                est_ai1, est_b1 = eval_est.affine_invariant_1(flow_pr_i, disp_gt_i)
                est_ai2, est_b2 = eval_est.affine_invariant_2(flow_pr_i, disp_gt_i)
                si, alpha = eval_est.scale_invariant(flow_pr_i, disp_gt_i)
                
                # Calculate binned EPE
                epe_per_bin, pixel_count_per_bin = eval_est.binned_epe(flow_pr_i, disp_gt_i, bins=quantile_edges)
                eval_est.add_binned_epe(epe_per_bin, pixel_count_per_bin, bin_edges=quantile_edges)
                
                est_ai2_fit = flow_pr_i * est_b2[0] + est_b2[1]
                
                # Get image path and create filename structure
                pth_lists = image_paths[0][i].split('/')
                pth = '/'.join(pth_lists[-2:])  # seq_340/image_name.png
                
                # Remove file extension to create directory structure
                image_name_no_ext = os.path.splitext(pth_lists[-1])[0]
                seq_dir = pth_lists[-2]  # seq_340
                
                # New path structure: seq_340/image_name/f_stop_X_X.png
                new_pth = os.path.join(seq_dir, image_name_no_ext, f'f_stop_{fstop_value}.png')
                
                filename = os.path.join(save_path.replace('result/train/', ''), new_pth)
                eval_est.add_filename(filename)

                val_id = i_batch * batch_size + i

                vrng = disp_gt_i.max() - disp_gt_i.min()
                vmargin = 0.1
                vmin, vmax = disp_gt_i.min() - vrng * vmargin, disp_gt_i.max() + vrng * vmargin
                eval_est.add_colorrange(vmin, vmax)

                if save_result:
                    # Create directories for the new path structure
                    for save_dir in [disp_dir, ai2_dir, ai2_fit_dir, epe_dir, epe0_3_dir, epe0_5_dir, ai2_0_3_dir, ai2_0_5_dir, gt_dir, src_dir]:
                        full_path = os.path.join(save_dir, new_pth)
                        os.makedirs(os.path.dirname(full_path), exist_ok=True)

                    # Save images with new naming structure
                    plt.imsave(os.path.join(disp_dir, new_pth), flow_pr_i.squeeze(), cmap='jet', vmin=vmin, vmax=vmax)
                    plt.imsave(os.path.join(ai2_fit_dir, new_pth), est_ai2_fit.squeeze(), cmap='jet', vmin=vmin, vmax=vmax)

                    err_rat = 0.7
                    vmin_err, vmax_err = 0, vrng * err_rat
                    plt.imsave(os.path.join(epe_dir, new_pth), np.abs(flow_pr_i.squeeze() - disp_gt_i.squeeze()), cmap='jet', vmin=vmin_err, vmax=vmax_err)
                    plt.imsave(os.path.join(ai2_dir, new_pth), np.abs(est_ai2_fit.squeeze() - disp_gt_i.squeeze()), cmap='jet', vmin=vmin_err, vmax=vmax_err)

                    err_rat_0_3 = 0.3
                    vmin_err_0_3, vmax_err_0_3 = 0, vrng * err_rat_0_3
                    plt.imsave(os.path.join(epe0_3_dir, new_pth), np.abs(flow_pr_i.squeeze() - disp_gt_i.squeeze()), cmap='jet', vmin=vmin_err_0_3, vmax=vmax_err_0_3)
                    plt.imsave(os.path.join(ai2_0_3_dir, new_pth), np.abs(est_ai2_fit.squeeze() - disp_gt_i.squeeze()), cmap='jet', vmin=vmin_err_0_3, vmax=vmax_err_0_3)

                    err_rat_0_5 = 0.5
                    vmin_err_0_5, vmax_err_0_5 = 0, vrng * err_rat_0_5
                    plt.imsave(os.path.join(epe0_5_dir, new_pth), np.abs(flow_pr_i.squeeze() - disp_gt_i.squeeze()), cmap='jet', vmin=vmin_err_0_5, vmax=vmax_err_0_5)
                    plt.imsave(os.path.join(ai2_0_5_dir, new_pth), np.abs(est_ai2_fit.squeeze() - disp_gt_i.squeeze()), cmap='jet', vmin=vmin_err_0_5, vmax=vmax_err_0_5)

                    plt.imsave(os.path.join(gt_dir, new_pth), disp_gt_i.squeeze(), cmap='jet', vmin=vmin, vmax=vmax)
                    plt.imsave(os.path.join(src_dir, new_pth), center_i.astype(np.uint8))

        # Save metrics for this f-stop
        eval_est.save_metrics()
        eval_est.save_binned_epe()
        eval_est.plot_binned_epe_histogram()
        
        # Get metrics for this f-stop
        fstop_result = eval_est.get_mean_metrics()
        fstop_metrics[fstop_value] = fstop_result
        
        # Add to results with f-stop prefix
        for key, value in fstop_result.items():
            all_results[f'f_stop_{fstop_value}/{key}'] = value
        
        print(f"F-stop {fstop_value} completed - EPE: {fstop_result.get('epe', 'N/A'):.4f}")

    # Calculate average metrics across all f-stops
    if fstop_metrics:
        avg_metrics = {}
        metric_keys = list(next(iter(fstop_metrics.values())).keys())
        
        for metric_key in metric_keys:
            values = [metrics[metric_key] for metrics in fstop_metrics.values() if metric_key in metrics]
            if values:
                avg_metrics[metric_key] = sum(values) / len(values)
        
        # Add average metrics to results
        for key, value in avg_metrics.items():
            all_results[f'avg/{key}'] = value
        
        print(f"\nOverall average EPE: {avg_metrics.get('epe', 'N/A'):.4f}")
        print(f"Processed {len(fstop_dirs)} f-stop datasets")

    return all_results


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

    # val_loader = data.DataLoader(val_dataset, batch_size=batch_size, 
    #     pin_memory=True, num_workers=0, drop_last=False)

    
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
    
    # Based on adaptive binning results with 15 bins, good pixel distribution:
    # Bin 1: [-1.500000, -1.125000] -> 2,718,528 pixels
    # Bin 2: [-1.125000, -0.750000] -> 4,248,316 pixels
    # Bin 3: [-0.750000, -0.562500] -> 3,632,914 pixels
    # Bin 4: [-0.562500, -0.375000] -> 1,311,188 pixels
    # Bin 5: [-0.375000, -0.281250] -> 3,882,535 pixels
    # Bin 6: [-0.281250, -0.187500] -> 3,248,622 pixels
    # Bin 7: [-0.187500, 0.005859] -> 28,604,280 pixels
    # Bin 8: [0.005859, 0.251953] -> 10,328,679 pixels
    # Bin 9: [0.251953, 0.333984] -> 19,316,021 pixels
    # Bin 10: [0.333984, 0.486328] -> 9,131,722 pixels
    # Bin 11: [0.486328, 0.597656] -> 19,152,237 pixels
    # Bin 12: [0.597656, 0.750000] -> 4,065,737 pixels
    # Bin 13: [0.750000, 0.843750] -> 2,759,007 pixels
    # Bin 14: [0.843750, 0.937500] -> 3,889,977 pixels
    # Bin 15: [0.937500, 1.125000] -> 202,116 pixels
    
    quantile_edges = np.array([-1.5, -1.125, -0.75, -0.5625, -0.375, -0.28125, -0.1875, 0.005859, 0.251953, 0.333984, 0.486328, 0.597656, 0.75, 0.84375, 0.9375, 1.125])
    
    # Set bin edges for the Eval object
    eval_est.bin_edges = quantile_edges
    print(f"Set {len(quantile_edges)-1} bin edges for binned EPE evaluation")
    
    for i_batch, data_blob in enumerate(tqdm(val_loader)):

        if i_batch % val_save_skip != 0:
            continue

        # if i_batch > 100:
        #     break

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
            
            # Calculate binned EPE using quantile bins
            epe_per_bin, pixel_count_per_bin = eval_est.binned_epe(flow_pr_i, disp_gt_i, bins=quantile_edges)
            eval_est.add_binned_epe(epe_per_bin, pixel_count_per_bin, bin_edges=quantile_edges)
            
            est_ai2_fit = flow_pr_i * est_b2[0] + est_b2[1]
            
            pth_lists = image_paths[0][i].split('/')
            pth = '/'.join(pth_lists[-2:])
            filename = os.path.join(save_path.replace('result/train/', ''), pth) 
            eval_est.add_filename(filename)
            # result[f'img/{val_id}/est_ai2_fit'] = est_ai2_fit
            # result[f'img/{val_id}/est'] = flow_pr_i
            # result[f'img/{val_id}/gt'] = disp_gt_i

            val_id = i_batch * batch_size + i

            vrng = disp_gt_i.max() - disp_gt_i.min()
            vmargin = 0.1
            vmin, vmax = disp_gt_i.min() - vrng * vmargin, disp_gt_i.max() + vrng * vmargin
            eval_est.add_colorrange(vmin, vmax)

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

                # img_est = Image.open(os.path.join(ai2_fit_dir, pth)).convert("RGB")
                # img_est = np.array(img_est)
                # img_est = np.moveaxis(img_est, -1, 0)
                # result[f'img/{val_id}/est'] = img_est
                # img_gt = Image.open(os.path.join(gt_dir, pth)).convert("RGB")
                # img_gt = np.array(img_gt)
                # img_gt = np.moveaxis(img_gt, -1, 0)
                # result[f'img/{val_id}/gt'] = np.array(img_gt)
                # img_src = Image.open(os.path.join(src_dir, pth)).convert("RGB")
                # img_src = np.array(img_src)
                # img_src = np.moveaxis(img_src, -1, 0)
                # result[f'img/{val_id}/src'] = np.array(img_src)

    eval_est.save_metrics()
    eval_est.save_binned_epe()  # Save binned EPE results
    eval_est.plot_binned_epe_histogram()  # Plot binned EPE histogram
    result = {**result, **eval_est.get_mean_metrics()}

    return result


@torch.no_grad()
def make_QPD(model, datatype='dual', gt_types=['disp', 'AiF'], iters=32, mixed_prec=False, save_result=False, val_save_skip=1, image_set='test', path='', save_path='result/train', batch_size=1, preprocess_params={'crop_h':672, 'crop_w':896, 'resize_h': 672, 'resize_w':896}, aug_params={}):
    # left, right, center, AiF, GT disparity, estimated disparity, correlation volume
    model.eval()

    if path == '':
        val_dataset = datasets.QPD(datatype=datatype, gt_types=gt_types, aug_params=aug_params, image_set=image_set, preprocess_params=preprocess_params)
    else:
        val_dataset = datasets.QPD(datatype=datatype, gt_types=gt_types, aug_params=aug_params, image_set=image_set, preprocess_params=preprocess_params, root=path)

    #FIXME : uncomment
    # val_loader = data.DataLoader(val_dataset, batch_size=batch_size, 
    #     pin_memory=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, drop_last=False)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, 
        pin_memory=True, num_workers=0, drop_last=False)
    
    
    path = os.path.basename(os.path.dirname(path))
    for i_batch, data_blob in enumerate(tqdm(val_loader)):
        if i_batch % val_save_skip != 0:
            continue

        image_paths = data_blob['image_list']
        center = data_blob['center'].cuda()
        lrtb_list = data_blob['lrtb_list'].cuda()
        disp_gt =  data_blob['disp'].cuda()
        valid_gt = data_blob['disp_valid'].cuda()
        aif = data_blob['AiF'].cuda()

        concat_lr = torch.cat([lrtb_list[:,0],lrtb_list[:,1]], dim=0).contiguous()
        
        with autocast(enabled=mixed_prec):
            _, flow_pr = model(center, concat_lr, iters=iters, test_mode=True)
            # _, (flow_pr, corr) = model(center, concat_lr, iters=iters, test_mode=True)

        flow_pr = flow_pr.cpu().numpy()
        disp_gt = disp_gt.cpu().numpy()
        center = center.permute(0,2,3,1).cpu().numpy()
        aif = aif.permute(0,2,3,1).cpu().numpy()
        left = lrtb_list[:,0].permute(0,2,3,1).cpu().numpy()
        right = lrtb_list[:,1].permute(0,2,3,1).cpu().numpy()
        # corr_cl = corr[0].cpu().numpy()
        # corr_cr = corr[1].cpu().numpy()
        
        disp_gt = disp_gt / 2

        assert flow_pr.shape == disp_gt.shape, (flow_pr.shape, disp_gt.shape)

        current_batch_size = flow_pr.shape[0]
        for i in range(current_batch_size):
            flow_pr_i = flow_pr[i]
            disp_gt_i = disp_gt[i]
            center_i = center[i]
            left_i = left[i]
            right_i = right[i]
            aif_i = aif[i]
            # corr_cl_i = corr_cl[i]
            # corr_cr_i = corr_cr[i]

            val_id = i_batch * batch_size + i

            if save_result:
                if not os.path.exists('result/predictions/'+path+'/'):
                    os.makedirs('result/predictions/'+path+'/')
            
                # pth_lists = image_paths[0][i].split('/')[-2:]
                pth_lists = image_paths[0][i].split('/')[-4:]
                center_pth = '/'.join(pth_lists)

                AiF_pth_lists = deepcopy(pth_lists)
                AiF_pth_lists[1] = 'target'
                AiF_pth = '/'.join(AiF_pth_lists)

                gt_disp_pth_lists = deepcopy(pth_lists)
                gt_disp_pth_lists[1] = 'target_disp'
                gt_disp_pth_lists[3] = gt_disp_pth_lists[3].replace('png', 'npy')
                gt_disp_pth = '/'.join(gt_disp_pth_lists)

                est_disp_pth_lists = deepcopy(pth_lists)
                est_disp_pth_lists[1] = 'FMDP_disp'
                est_disp_pth_lists[3] = est_disp_pth_lists[3].replace('png', 'npy')
                est_disp_pth = '/'.join(est_disp_pth_lists)

                # corr_cl_pth_lists = deepcopy(pth_lists)
                # # corr_cl_pth_lists[1] = 'corr_cl'
                # # corr_cl_pth_lists[3] = corr_cl_pth_lists[3].replace('png', 'npy')
                # # corr_cl_pth = '/'.join(corr_cl_pth_lists)

                # corr_cr_pth_lists = deepcopy(pth_lists)
                # # corr_cr_pth_lists[1] = 'corr_cr'
                # # corr_cr_pth_lists[3] = corr_cr_pth_lists[3].replace('png', 'npy')
                # # corr_cr_pth = '/'.join(corr_cr_pth_lists)

                pth_lists = image_paths[1][i].split('/')[-4:]
                left_pth = '/'.join(pth_lists)

                pth_lists = image_paths[2][i].split('/')[-4:]
                right_pth = '/'.join(pth_lists)

                # paths = [center_pth, AiF_pth, gt_disp_pth, est_disp_pth, corr_cl_pth, corr_cr_pth, left_pth, right_pth]
                paths = [center_pth, AiF_pth, gt_disp_pth, est_disp_pth, left_pth, right_pth]
                for i in range(len(paths)):
                    paths[i] = os.path.join(save_path, paths[i])
                    os.makedirs(os.path.dirname(paths[i]), exist_ok=True)
                
                # center_pth, AiF_pth, gt_disp_pth, est_disp_pth, corr_cl_pth, corr_cr_pth, left_pth, right_pth = paths
                center_pth, AiF_pth, gt_disp_pth, est_disp_pth, left_pth, right_pth = paths

                plt.imsave(center_pth, center_i.astype(np.uint8))
                plt.imsave(left_pth, left_i.astype(np.uint8))
                plt.imsave(right_pth, right_i.astype(np.uint8))
                plt.imsave(AiF_pth, aif_i.astype(np.uint8))

                np.save(gt_disp_pth, disp_gt_i.squeeze())
                np.save(est_disp_pth, flow_pr_i.squeeze())

                est_disp_png_pth = est_disp_pth.replace('FMDP_disp', 'FMDP_disp_png')[:-4] + '.png'
                os.makedirs(os.path.dirname(est_disp_png_pth), exist_ok=True)
                plt.imsave(est_disp_png_pth, flow_pr_i.squeeze())


@torch.no_grad()
def make_DDDP(model, datatype='dual', gt_types=['AiF'], iters=32, mixed_prec=False, save_result=False, val_save_skip=1, image_set='test', path='', save_path='result/train', batch_size=1, preprocess_params={'crop_h':672, 'crop_w':896, 'resize_h': 672, 'resize_w':896}, aug_params={}):
    # left, right, center, AiF, GT disparity, estimated disparity, correlation volume
    model.eval()

    if path == '':
        val_dataset = datasets.DDDP(datatype=datatype, gt_types=gt_types, aug_params=aug_params, image_set=image_set, preprocess_params=preprocess_params)
    else:
        val_dataset = datasets.DDDP(datatype=datatype, gt_types=gt_types, aug_params=aug_params, image_set=image_set, preprocess_params=preprocess_params, root=path)

    #FIXME : uncomment
    # val_loader = data.DataLoader(val_dataset, batch_size=batch_size, 
    #     pin_memory=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, drop_last=False)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, 
        pin_memory=True, num_workers=0, drop_last=False)
    
    
    path = os.path.basename(os.path.dirname(path))
    for i_batch, data_blob in enumerate(tqdm(val_loader)):
        if i_batch % val_save_skip != 0:
            continue

        image_paths = data_blob['image_list']
        center = data_blob['center'].cuda()
        lrtb_list = data_blob['lrtb_list'].cuda()
        # disp_gt =  data_blob['disp'].cuda()
        # valid_gt = data_blob['disp_valid'].cuda()
        aif = data_blob['AiF'].cuda()

        concat_lr = torch.cat([lrtb_list[:,0],lrtb_list[:,1]], dim=0).contiguous()
        
        with autocast(enabled=mixed_prec):
            _, flow_pr = model(center, concat_lr, iters=iters, test_mode=True)
            # _, (flow_pr, corr) = model(center, concat_lr, iters=iters, test_mode=True)

        flow_pr = flow_pr.cpu().numpy()
        # disp_gt = disp_gt.cpu().numpy()
        center = center.permute(0,2,3,1).cpu().numpy()
        aif = aif.permute(0,2,3,1).cpu().numpy()
        left = lrtb_list[:,0].permute(0,2,3,1).cpu().numpy()
        right = lrtb_list[:,1].permute(0,2,3,1).cpu().numpy()
        # corr_cl = corr[0].cpu().numpy()
        # corr_cr = corr[1].cpu().numpy()
        
        # # disp_gt = disp_gt / 2

        # # assert flow_pr.shape == disp_gt.shape, (flow_pr.shape, disp_gt.shape)

        current_batch_size = flow_pr.shape[0]
        for i in range(current_batch_size):
            flow_pr_i = flow_pr[i]
            # # disp_gt_i = disp_gt[i]
            center_i = center[i]
            left_i = left[i]
            right_i = right[i]
            aif_i = aif[i]
            # corr_cl_i = corr_cl[i]
            # corr_cr_i = corr_cr[i]

            val_id = i_batch * batch_size + i

            if save_result:
                if not os.path.exists('result/predictions/'+path+'/'):
                    os.makedirs('result/predictions/'+path+'/')
            
                # pth_lists = image_paths[0][i].split('/')[-2:]
                pth_lists = image_paths[0][i].split('/')[-4:]
                center_pth = '/'.join(pth_lists)

                AiF_pth_lists = deepcopy(pth_lists)
                AiF_pth_lists[2] = 'target'
                AiF_pth = '/'.join(AiF_pth_lists)

                gt_disp_pth_lists = deepcopy(pth_lists)
                gt_disp_pth_lists[2] = 'target_disp'
                # gt_disp_pth_lists = gt_disp_pth_lists[:2] + ['target_disp'] + gt_disp_pth_lists[2:]
                gt_disp_pth_lists[3] = gt_disp_pth_lists[3].replace('png', 'npy')
                gt_disp_pth = '/'.join(gt_disp_pth_lists)

                est_disp_pth_lists = deepcopy(pth_lists)
                est_disp_pth_lists[2] = 'FMDP_disp'
                # est_disp_pth_lists = est_disp_pth_lists[:2] + ['target_disp'] + est_disp_pth_lists[2:]
                est_disp_pth_lists[3] = est_disp_pth_lists[3].replace('png', 'npy')
                est_disp_pth = '/'.join(est_disp_pth_lists)

                # corr_cl_pth_lists = deepcopy(pth_lists)
                # # corr_cl_pth_lists[1] = 'corr_cl'
                # # corr_cl_pth_lists[3] = corr_cl_pth_lists[3].replace('png', 'npy')
                # # corr_cl_pth = '/'.join(corr_cl_pth_lists)

                # corr_cr_pth_lists = deepcopy(pth_lists)
                # # corr_cr_pth_lists[1] = 'corr_cr'
                # # corr_cr_pth_lists[3] = corr_cr_pth_lists[3].replace('png', 'npy')
                # # corr_cr_pth = '/'.join(corr_cr_pth_lists)

                pth_lists = image_paths[1][i].split('/')[-4:]
                left_pth = '/'.join(pth_lists)

                pth_lists = image_paths[2][i].split('/')[-4:]
                right_pth = '/'.join(pth_lists)

                # paths = [center_pth, AiF_pth, gt_disp_pth, est_disp_pth, corr_cl_pth, corr_cr_pth, left_pth, right_pth]
                paths = [center_pth, AiF_pth, gt_disp_pth, est_disp_pth, left_pth, right_pth]
                for i in range(len(paths)):
                    paths[i] = os.path.join(save_path, paths[i])
                    os.makedirs(os.path.dirname(paths[i]), exist_ok=True)
                
                # center_pth, AiF_pth, gt_disp_pth, est_disp_pth, corr_cl_pth, corr_cr_pth, left_pth, right_pth = paths
                center_pth, AiF_pth, gt_disp_pth, est_disp_pth, left_pth, right_pth = paths

                plt.imsave(center_pth, center_i.astype(np.uint8))
                plt.imsave(left_pth, left_i.astype(np.uint8))
                plt.imsave(right_pth, right_i.astype(np.uint8))
                plt.imsave(AiF_pth, aif_i.astype(np.uint8))

                # np.save(gt_disp_pth, disp_gt_i.squeeze())
                np.save(est_disp_pth, flow_pr_i.squeeze())

                # print(f"{flow_pr_i.min()} {flow_pr_i.max()}")

                est_disp_png_pth = est_disp_pth.replace('FMDP_disp', 'FMDP_disp_png')[:-4] + '.png'
                os.makedirs(os.path.dirname(est_disp_png_pth), exist_ok=True)
                plt.imsave(est_disp_png_pth, flow_pr_i.squeeze(), vmin=-3.5, vmax=1.5)



if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='Interp', help="name your experiment")
    parser.add_argument('--ckpt_epoch', type=str, default=0)
    parser.add_argument('--save_result', action='store_true', help="Save predicted results")
    parser.add_argument('--eval_datasets', choices=['QPD-Test', 'QPD-TransDisk-Test', 'QPD-FStop-1_2-Test', 'QPD-FStop-1_4-Test', 'QPD-FStop-2_0-Test', 'QPD-FStop-2_8-Test', 'QPD-FStops-All', 'QPD-Valid', 'DPD_Disp', 'Real_QPD', 'QPD-Test-noise', 'DP5K-Valid', 'DP5K-Test', 'DP5K-Test-Lowres', 'DP119', 'Make_QPD', 'Make_DDDP'], nargs='+', default=[], required=True, help="Additional dataset to evaluate")

    args = parser.parse_args()

    # conf = get_train_config(args.exp_name)
    conf = get_run_setting(args.exp_name)


    if args.ckpt_epoch == 'latest':
        restore_ckpt = os.path.join(conf.save_path, 'checkpoints', 'latest.pth')
    else:
        args.ckpt_epoch = int(args.ckpt_epoch)
        ckpts = get_ckpts_in_dir(conf.save_path) # Get all checkpoints sorted by epoch
        for ckpt in ckpts:
            try:
                epoch = int(os.path.basename(ckpt).split('_')[0])
            except Exception as e:
                print(f'{e} occured from ckpt: {ckpt}')

            if epoch == args.ckpt_epoch: # Find the specified epoch
                restore_ckpt = ckpt
                break

    model = MonoQPD(conf)
    if restore_ckpt is not None:
        assert str(restore_ckpt).endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(restore_ckpt)
        model.da_v2.load_state_dict(torch.load('mono_qpd/Depth_Anything_V2/checkpoints/depth_anything_v2_vitl.pth'))
        if 'qpdnet_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint and 'scheduler_state_dict' in checkpoint:
            c={}
            c['qpdnet_state_dict'] = fix_key(checkpoint['qpdnet_state_dict'])
            model.qpdnet.load_state_dict(c['qpdnet_state_dict'])
            model.feature_converter.load_state_dict(fix_key(checkpoint['fcvt_state_dict']))
            epoch = checkpoint['epoch']

        # # For loading old checkpoints
        # model.load_state_dict(checkpoint, strict=True)
        # if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint and 'scheduler_state_dict' in checkpoint:
            # model_dummy = MonoQPD(conf)
            # model_dummy.load_state_dict(checkpoint['model_state_dict'], strict=True)
            # c['qpdnet_state_dict'] = fix_key(model_dummy.qpdnet.state_dict())
            # c['fcvt_state_dict'] = fix_key(model_dummy.feature_converter.state_dict())
            # model.qpdnet.load_state_dict(c['qpdnet_state_dict'], strict=True)
            # model.feature_converter.load_state_dict(c['fcvt_state_dict'], strict=True)
            
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
        save_dir = os.path.join(conf.save_path, 'qpd-test')
        save_path = os.path.join(save_dir, f'{epoch:03d}_epoch')
        print(save_path)
        result = validate_QPD(model, iters=conf.valid_iters, mixed_prec=use_mixed_precision, save_result=True if args.save_result else False, datatype = conf.datatype, image_set="test", path='datasets/QP-Data', save_path=save_path, batch_size=conf.qpd_test_bs if conf.qpd_test_bs else 1)

        log_dir = os.path.join(save_dir, 'runs')
        logger = EvalLogger(log_dir=log_dir, epoch=epoch)

        named_results = {}
        for k, v in result.items():
            named_results[f'test_qpd/{k}'] = v
            if 'img' not in k:
                print(f'test_qpd/{k}: {v}')

        logger.write_dict(named_results)

    if 'QPD-TransDisk-Test' in args.eval_datasets:
        save_dir = os.path.join(conf.save_path, 'qpd-transdisk-test')
        save_path = os.path.join(save_dir, f'{epoch:03d}_epoch')
        print(save_path)
        result = validate_QPD(model, iters=conf.valid_iters, mixed_prec=use_mixed_precision, save_result=True if args.save_result else False, datatype = conf.datatype, image_set="test", path='datasets/QP-Data-TransDisk', save_path=save_path, batch_size=conf.qpd_test_bs if conf.qpd_test_bs else 1)

        log_dir = os.path.join(save_dir, 'runs')
        logger = EvalLogger(log_dir=log_dir, epoch=epoch)

        named_results = {}
        for k, v in result.items():
            named_results[f'test_qpd_transdisk/{k}'] = v
            if 'img' not in k:
                print(f'test_qpd_transdisk/{k}: {v}')

        logger.write_dict(named_results)

    if 'QPD-FStop-1_2-Test' in args.eval_datasets:
        save_dir = os.path.join(conf.save_path, 'qpd-fstop-1_2-test')
        save_path = os.path.join(save_dir, f'{epoch:03d}_epoch')
        print(save_path)
        result = validate_QPD(model, iters=conf.valid_iters, mixed_prec=use_mixed_precision, save_result=True if args.save_result else False, datatype = conf.datatype, image_set="test", path='datasets/QP-Data-FStop-1_2', save_path=save_path, batch_size=conf.qpd_test_bs if conf.qpd_test_bs else 1)

        log_dir = os.path.join(save_dir, 'runs')
        logger = EvalLogger(log_dir=log_dir, epoch=epoch)

        named_results = {}
        for k, v in result.items():
            named_results[f'test_qpd_fstop_1_2/{k}'] = v
            if 'img' not in k:
                print(f'test_qpd_fstop_1_2/{k}: {v}')

        logger.write_dict(named_results)

    if 'QPD-FStop-1_4-Test' in args.eval_datasets:
        save_dir = os.path.join(conf.save_path, 'qpd-fstop-1_4-test')
        save_path = os.path.join(save_dir, f'{epoch:03d}_epoch')
        print(save_path)
        result = validate_QPD(
            model,
            iters=conf.valid_iters,
            mixed_prec=use_mixed_precision,
            save_result=True if args.save_result else False,
            datatype=conf.datatype,
            image_set="test",
            path='datasets/QP-Data-FStop-1_4',
            save_path=save_path,
            batch_size=conf.qpd_test_bs if conf.qpd_test_bs else 1
        )

        log_dir = os.path.join(save_dir, 'runs')
        logger = EvalLogger(log_dir=log_dir, epoch=epoch)

        named_results = {}
        for k, v in result.items():
            named_results[f'test_qpd_fstop_1_4/{k}'] = v
            if 'img' not in k:
                print(f'test_qpd_fstop_1_4/{k}: {v}')

        logger.write_dict(named_results)

    if 'QPD-FStop-2_0-Test' in args.eval_datasets:
        save_dir = os.path.join(conf.save_path, 'qpd-fstop-2_0-test')
        save_path = os.path.join(save_dir, f'{epoch:03d}_epoch')
        print(save_path)
        result = validate_QPD(
            model,
            iters=conf.valid_iters,
            mixed_prec=use_mixed_precision,
            save_result=True if args.save_result else False,
            datatype=conf.datatype,
            image_set="test",
            path='datasets/QP-Data-FStop-2_0',
            save_path=save_path,
            batch_size=conf.qpd_test_bs if conf.qpd_test_bs else 1
        )

        log_dir = os.path.join(save_dir, 'runs')
        logger = EvalLogger(log_dir=log_dir, epoch=epoch)

        named_results = {}
        for k, v in result.items():
            named_results[f'test_qpd_fstop_2_0/{k}'] = v
            if 'img' not in k:
                print(f'test_qpd_fstop_2_0/{k}: {v}')

        logger.write_dict(named_results)

    if 'QPD-FStop-2_8-Test' in args.eval_datasets:
        save_dir = os.path.join(conf.save_path, 'qpd-fstop-2_8-test')
        save_path = os.path.join(save_dir, f'{epoch:03d}_epoch')
        print(save_path)
        result = validate_QPD(
            model,
            iters=conf.valid_iters,
            mixed_prec=use_mixed_precision,
            save_result=True if args.save_result else False,
            datatype=conf.datatype,
            image_set="test",
            path='datasets/QP-Data-FStop-2_8',
            save_path=save_path,
            batch_size=conf.qpd_test_bs if conf.qpd_test_bs else 1
        )

        log_dir = os.path.join(save_dir, 'runs')
        logger = EvalLogger(log_dir=log_dir, epoch=epoch)

        named_results = {}
        for k, v in result.items():
            named_results[f'test_qpd_fstop_2_8/{k}'] = v
            if 'img' not in k:
                print(f'test_qpd_fstop_2_8/{k}: {v}')

        logger.write_dict(named_results)

    if 'QPD-FStops-All' in args.eval_datasets:
        save_dir = os.path.join(conf.save_path, 'qpd-fstops')
        save_path = os.path.join(save_dir, f'{epoch:03d}_epoch')
        print(save_path)
        result = validate_QPD_FStops(
            model,
            iters=conf.valid_iters,
            mixed_prec=use_mixed_precision,
            save_result=True if args.save_result else False,
            datatype=conf.datatype,
            image_set="test",
            path='datasets/qpd-test-fstops',
            save_path=save_path,
            batch_size=conf.qpd_test_bs if conf.qpd_test_bs else 1
        )

        log_dir = os.path.join(save_dir, 'runs')
        logger = EvalLogger(log_dir=log_dir, epoch=epoch)

        named_results = {}
        for k, v in result.items():
            named_results[f'test_qpd_seq_340/{k}'] = v
            if 'img' not in k:
                print(f'test_qpd_seq_340/{k}: {v}')

        logger.write_dict(named_results)

    if 'QPD-Test-noise' in args.eval_datasets:
        save_dir = os.path.join(conf.save_path, 'qpd-test-noise')
        save_path = os.path.join(save_dir, f'{epoch:03d}_epoch')
        print(save_path)
        result = validate_QPD(model, iters=conf.valid_iters, mixed_prec=use_mixed_precision, save_result=True if args.save_result else False, datatype = conf.datatype, image_set="test", path='datasets/QP-Data-noise0.001', save_path=save_path, batch_size=conf.qpd_test_bs if conf.qpd_test_bs else 1)
        
        log_dir = os.path.join(save_dir, 'runs') 
        logger = EvalLogger(log_dir=log_dir, epoch=epoch)
        
        named_result = {}
        for k, v in result.items():
            named_result[f'val_qpd_test_noise/{k}'] = v
            if 'img' not in k:
                print(f'val_qpd_test_noise/{k}: {v}')

        logger.write_dict(named_result)
        logger.close()

    if 'QPD-Valid' in args.eval_datasets:
        save_dir = os.path.join(conf.save_path, 'qpd-valid')
        save_path = os.path.join(save_dir, f'{epoch:03d}_epoch')
        print(save_path)

        result = validate_QPD(model, iters=conf.valid_iters, mixed_prec=use_mixed_precision, save_result=True if args.save_result else False, datatype = conf.datatype, image_set="validation", path='datasets/QP-Data', save_path=save_path, batch_size=conf.qpd_valid_bs if conf.qpd_valid_bs else 1)
        # result = {}
        
        log_dir = os.path.join(save_dir, 'runs') 
        logger = EvalLogger(log_dir=log_dir, epoch=epoch) # epoch=checkpoint['total_steps'])
        
        named_result = {}
        for k, v in result.items():
            named_result[f'val_qpd_valid/{k}'] = v
            if 'img' not in k:
                print(f'val_qpd_valid/{k}: {v}')

        logger.write_dict(named_result)
        logger.close()

    if 'DPD_Disp' in args.eval_datasets:
        save_dir = os.path.join(conf.save_path, 'dp-disp')
        save_path = os.path.join(save_dir, f'{epoch:03d}_epoch')
        print(save_path)
        result = validate_DPD_Disp(model, iters=conf.valid_iters, mixed_prec=use_mixed_precision, save_result=True if args.save_result else False, datatype = conf.datatype, image_set="test", path='datasets/MDD_dataset', save_path=save_path, batch_size=conf.dp_disp_bs if conf.dp_disp_bs else 1)
        
        log_dir = os.path.join(save_dir, 'runs') 
        logger = EvalLogger(log_dir=log_dir, epoch=epoch)
        
        named_result = {}
        for k, v in result.items():
            named_result[f'val_qpd_dpd_disp/{k}'] = v
            if 'img' not in k:
                print(f'val_qpd_dpd_disp/{k}: {v}')

        logger.write_dict(named_result)
        logger.close()
    
    if 'DP119' in args.eval_datasets:
        save_dir = os.path.join(conf.save_path, 'dp119')
        save_path = os.path.join(save_dir, f'{epoch:03d}_epoch')
        print(save_path)
        result = validate_DP119(model, iters=conf.valid_iters, mixed_prec=use_mixed_precision, save_result=True if args.save_result else False, datatype = conf.datatype, image_set="test", path='datasets/DP119', save_path=save_path, batch_size=conf.dp_disp_bs if conf.dp_disp_bs else 1)
        
        log_dir = os.path.join(save_dir, 'runs') 
        logger = EvalLogger(log_dir=log_dir, epoch=epoch)
        
        named_result = {}
        for k, v in result.items():
            named_result[f'dp119/{k}'] = v
            if 'img' not in k:
                print(f'dp119/{k}: {v}')

        logger.write_dict(named_result)
        logger.close()


    if 'DP5K-Test' in args.eval_datasets:
        save_dir = os.path.join(conf.save_path, 'dp5k-test')
        save_path = os.path.join(save_dir, f'{epoch:03d}_epoch')
        print(save_path)
        result = validate_DP5K(model, iters=conf.valid_iters, mixed_prec=use_mixed_precision, save_result=True if args.save_result else False, datatype = conf.datatype, image_set="test", path='datasets/DP5K', save_path=save_path, batch_size=conf.dp_disp_bs if conf.dp_disp_bs else 1)
        
        log_dir = os.path.join(save_dir, 'runs') 
        logger = EvalLogger(log_dir=log_dir, epoch=epoch)
        
        named_result = {}
        for k, v in result.items():
            named_result[f'test_dp5k/{k}'] = v
            if 'img' not in k:
                print(f'test_dp5k/{k}: {v}')

        logger.write_dict(named_result)
        logger.close()

    if 'DP5K-Test-Lowres' in args.eval_datasets:
        save_dir = os.path.join(conf.save_path, 'dp5k-test-lowres')
        save_path = os.path.join(save_dir, f'{epoch:03d}_epoch')
        print(save_path)
        result = validate_DP5K(model, iters=conf.valid_iters, mixed_prec=use_mixed_precision, save_result=True if args.save_result else False, datatype = conf.datatype, image_set="test", path='datasets/DP5K', save_path=save_path, batch_size=conf.dp_disp_bs if conf.dp_disp_bs else 1, preprocess_params={'crop_h':1120, 'crop_w':1120, 'resize_h':896, 'resize_w':896})
        
        log_dir = os.path.join(save_dir, 'runs') 
        logger = EvalLogger(log_dir=log_dir, epoch=epoch)
        
        named_result = {}
        for k, v in result.items():
            named_result[f'test_lowres_dp5k/{k}'] = v
            if 'img' not in k:
                print(f'test_lowres_dp5k/{k}: {v}')

        logger.write_dict(named_result)
        logger.close()

    if 'DP5K-Valid' in args.eval_datasets:
        save_dir = os.path.join(conf.save_path, 'dp5k-valid')
        save_path = os.path.join(save_dir, f'{epoch:03d}_epoch')
        print(save_path)
        result = validate_DP5K(model, iters=conf.valid_iters, mixed_prec=use_mixed_precision, save_result=True if args.save_result else False, datatype = conf.datatype, image_set="valid", path='datasets/DP5K', save_path=save_path, batch_size=conf.dp_disp_bs if conf.dp_disp_bs else 1)
        
        log_dir = os.path.join(save_dir, 'runs') 
        logger = EvalLogger(log_dir=log_dir, epoch=epoch)
        
        named_result = {}
        for k, v in result.items():
            named_result[f'val_dp5k/{k}'] = v
            if 'img' not in k:
                print(f'val_dp5k/{k}: {v}')

        logger.write_dict(named_result)
        logger.close()

    if 'Real_QPD' in args.eval_datasets:
        save_dir = os.path.join(conf.save_path, 'real-qpd-test')
        save_path = os.path.join(save_dir, f'{epoch:03d}_epoch')
        print(save_path)
        result = validate_Real_QPD(model, iters=conf.valid_iters, mixed_prec=use_mixed_precision, save_result=True if args.save_result else False, datatype = conf.datatype, image_set="test", path='datasets/Real-QP-Data', save_path=save_path, batch_size=conf.real_qpd_bs if conf.real_qpd_bs else 1)
        
        log_dir = os.path.join(save_dir, 'runs') 
        logger = EvalLogger(log_dir=log_dir, epoch=epoch)
        
        named_result = {}
        for k, v in result.items():
            named_result[f'val_real_qpd/{k}'] = v
            if 'img' not in k:
                print(f'val_real_qpd/{k}: {v}')

        logger.write_dict(named_result)
        logger.close()



    elapsed = time.time() - start_time

    print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(elapsed)))
    print(result)

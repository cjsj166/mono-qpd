from __future__ import print_function, division

import numpy
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
# from QPDNet.qpd_net import QPDNet
from mono_qpd.mono_qpd import MonoQPD
import os
from mono_qpd.loss import ScaleInvariantLoss, LeastSquareScaleInvariantLoss
# from evaluate_mono_qpd import *
import mono_qpd.QPDNet.Quad_datasets as datasets
from argparse import Namespace
from evaluate_mono_qpd import validate_QPD, validate_DPD_Disp
from datetime import datetime
from logger import Logger

from runsync.presets import get_run_setting


try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


def sequence_loss(flow_preds, flow_gt, valid, si_loss_weight=0.0, loss_gamma=0.9, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    b,c,h,w = flow_gt.shape
    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    # exclude extremly large displacements
    valid = ((valid.squeeze(1) >= 0.5) & (mag < max_flow)).unsqueeze(1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any(), "Invalid values in flow predictions"
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)

        fp = flow_preds[i]

        si_loss = 0
        if si_loss_weight != 0:
            criterion = LeastSquareScaleInvariantLoss()
            si_loss = criterion(fp, (flow_gt/2), valid) * si_loss_weight

        l1_loss = 0
        if si_loss_weight != 1:    
            l1_loss = (fp-(flow_gt/2)).abs()
            assert l1_loss.shape == valid.shape, [l1_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
            l1_loss = l1_loss[valid.bool()].mean()

        i_loss = si_loss * si_loss_weight + l1_loss * (1 - si_loss_weight)
        flow_loss += i_weight * i_loss
    
    fp = flow_preds[-1]
    epe = torch.sum((fp - (flow_gt)/2)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '0.005px': (epe < 0.005).float().mean().item(),
        '0.01px': (epe < 0.01).float().mean().item(),
        '0.05px': (epe < 0.05).float().mean().item(),
    }

    return flow_loss, metrics


def fetch_optimizer(args, model, last_epoch=-1):
    """ Create the optimizer and learning rate scheduler """
    if last_epoch == -1:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
                pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    else:
        max_lr = args.lr
        optimizer = optim.AdamW([{'params': model.parameters(), 'initial_lr': max_lr, 'max_lr': args.lr, 
                                  'min_lr': 1e-8}], lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps = args.num_steps+100,
                pct_start=0.01, cycle_momentum=False, anneal_strategy='linear', last_epoch=last_epoch)

    return optimizer, scheduler

# Functions for NaN debugging
def check_nan(module, name, output):
    if isinstance(output, tuple) or isinstance(output, list):
        for o in output:
            check_nan(module, name, o)
    else:
        if torch.isnan(output).any():
            print(f"⚠ NaN detected in {name}")
            print(f"⚠ NaN detected in {module.__class__.__name__}")

def check_nan_hook(name):
    def check_nan_hook(module, input, output):
        check_nan(module, name, output)        
    return check_nan_hook

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(args):

    torch.manual_seed(1234)
    np.random.seed(1234)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')


    model = MonoQPD(args)
    print("Parameter Count: %d" % count_parameters(model))

    # Codes for debugging NaN
    for name, layer in model.named_modules():
        layer.register_forward_hook(check_nan_hook(name))

    # da_v2_args = args['da_v2']
    # args = args['else']

    # Prepare the save directory
    save_dir = os.path.join(args.save_path)
    model_save_dir = os.path.join(save_dir, 'checkpoints')
    log_dir = os.path.join(save_dir, 'runs')

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_loader = datasets.fetch_dataloader(args)
    total_steps = 0
    optimizer, scheduler = fetch_optimizer(args, model, -1)
    model.da_v2.load_state_dict(torch.load(args.restore_ckpt_da_v2))
    # if args.restore_ckpt_mono_qpd is not None:
    #     # assert os.path.exists(args.restore_ckpt_mono_qpd)
    if args.restore_ckpt_mono_qpd is not None and os.path.exists(args.restore_ckpt_mono_qpd):

        ckpt = torch.load(args.restore_ckpt_mono_qpd)
        total_steps = ckpt['total_steps']
        model.qpdnet.load_state_dict(ckpt['qpdnet_state_dict'])
        model.feature_converter.load_state_dict(ckpt['fcvt_state_dict'])

        model = nn.DataParallel(model)
        model.cuda()
    
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        if not args.initialize_scheduler:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    else:
        print(f"Checkpoint not found. Training from scratch.")
        model = nn.DataParallel(model)
        model.cuda()


    if args.freeze_da_v2:
        for param in model.module.da_v2.parameters():
            param.requires_grad = False
    
    if args.dec_update:
        for param in model.module.da_v2.depth_head.parameters():
            param.requires_grad = True

    # Save the arguments
    with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')
        f.write('\n')
        # for key, value in vars(da_v2_args).items():
        #     f.write(f'{key}: {value}\n')

    logger = Logger(model, scheduler, total_steps, log_dir=log_dir)

    model.train()
    # model.module.freeze_bn() # We keep BatchNorm frozen

    validation_frequency = 10000

    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True
    global_batch_num = total_steps
    batch_len = len(train_loader)
    epoch = int(total_steps/batch_len)

    qpd_epebest,qpd_rmsebest,qpd_ai2best = 1000,1000,1000
    qpd_epeepoch,qpd_rmseepoch,qpd_ai2epoch = 0,0,0
    dpdisp_epebest,dpdisp_rmsebest,dpdisp_ai2best = 1000,1000,1000
    dpdisp_epeepoch,dpdisp_rmseepoch,dpdisp_ai2epoch = 0,0,0

    while should_keep_training:
        for i_batch, data_blob in enumerate(tqdm(train_loader)):
            # if args.debug_mode:
            # if i_batch > 3:
            #     break

            optimizer.zero_grad()

            center_img = data_blob['center'].cuda()
            lrtblist = data_blob['lrtb_list'].cuda()
            flow = data_blob['disp'].cuda()
            valid = data_blob['disp_valid'].cuda()
            # center_img, lrtblist, flow, valid = [x.cuda() for x in data_blob]

            assert not torch.isnan(center_img).any(), "Invalid values in input images"
            assert not torch.isnan(lrtblist).any(), "Invalid values in input images"

            b,s,c,h,w = lrtblist.shape

            image1 = center_img.contiguous().view(b,c,h,w)
            if args.datatype == 'quad':
                image2 = torch.cat([lrtblist[:,0],lrtblist[:,1],lrtblist[:,2],lrtblist[:,3]], dim=0).contiguous()
            elif args.datatype == 'dual':
                image2 = torch.cat([lrtblist[:,0],lrtblist[:,1]], dim=0).contiguous()
            else:
                raise NotImplementedError

            assert model.training
            flow_predictions = model(image1, image2, iters=args.train_iters)
            assert model.training
            if args.input_image_num == 42:
                rot_flow_predictions=[]
                for i in range(len(flow_predictions)):
                    rot_flow_predictions.append(torch.rot90(flow_predictions[i], k=-1, dims=[2,3]))   
                loss, metrics = sequence_loss(rot_flow_predictions, flow, valid, si_loss_weight=args.si_loss)
            elif args.input_image_num == 24:
                rot_flow_predictions=[]
                for i in range(len(flow_predictions)):
                    rot_flow_predictions.append(torch.rot90(flow_predictions[i], k=1, dims=[2,3]))   
                loss, metrics = sequence_loss(rot_flow_predictions, flow, valid, si_loss_weight=args.si_loss)
            else:
                try:
                    loss, metrics = sequence_loss(flow_predictions, flow, valid, si_loss_weight=args.si_loss)
                    
                except AssertionError as e:
                    if "Invalid values in flow predictions" in str(e):
                        print(f"Invalid values in flow predictions, epoch: {epoch}, batch: {i_batch}")
                        continue
                    else:
                        raise e
                                    
            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)                        
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            total_steps += 1

            if total_steps % (batch_len // 8) == 0 or total_steps==1 or (args.stop_step is not None and total_steps >= args.stop_step):# and total_steps != 0:    
                # total_steps % (batch_len * 2)
                epoch = int(total_steps/batch_len)
                
                model_save_path = os.path.join(args.save_path, 'checkpoints', f'{epoch:03d}_epoch_{total_steps}_{args.name}.pth')
                model_save_path = Path(model_save_path).absolute()

                print(os.path.basename(model_save_path))
                logging.info(f"Saving file {model_save_path}")
                torch.save({
                            'qpdnet_state_dict': model.module.qpdnet.state_dict(),
                            'fcvt_state_dict': model.module.feature_converter.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'total_steps': total_steps,
                            'epoch': epoch,
                            # ... any other states you need
                            }, model_save_path)

                # Update the latest symlink
                tmp = os.path.join(model_save_dir, "latest.pth.tmp")
                latest = os.path.join(model_save_dir, "latest.pth")

                if os.path.islink(tmp) or os.path.exists(tmp): os.unlink(tmp)
                os.symlink(model_save_path, tmp)
                os.replace(tmp, latest)

            # validation function call, but commented as we use separate validation script
            # if total_steps % (batch_len*5) == 0 or total_steps==1:
            #     if total_steps == 1:
            #         val_save_skip = 50
                
            #     save_dir = os.path.join(args.save_path, 'qpd-valid', f'{epoch:03d}_epoch')

            #     # FIXME: This is a temporary fix for the bug in the validation code
            #     # commented temporalily ----------------------------------------------------------------------------------

            #     val_save_skip = 380 // 10
            #     if args.debug_mode:
            #         val_save_skip = 370
            #         # val_save_skip = 380 // 10

            #     results = validate_QPD(model.module, iters=args.valid_iters, save_result=True, val_save_skip=val_save_skip, datatype=args.datatype, image_set='validation', path='datasets/QP-Data', save_path=save_dir, batch_size=args.qpd_valid_bs)
                    
            #     if qpd_epebest>=results['epe']:
            #         qpd_epebest = results['epe']
            #         qpd_epeepoch = epoch
            #     if qpd_rmsebest>=results['rmse']:
            #         qpd_rmsebest = results['rmse']
            #         qpd_rmseepoch = epoch
            #     if qpd_ai2best>=results['ai2']:
            #         qpd_ai2best = results['ai2']
            #         qpd_ai2epoch = epoch
                
                
            #     named_results = {}
            #     for k, v in results.items():
            #         named_results[f'val_qpd/{k}'] = v
            #         if 'img' not in k:
            #             print(f'val_qpd/{k}: {v}')

            #     logger.write_dict(named_results)

            #     # logging.info(f"Current Best Result qpd epe epoch {qpd_epeepoch}, result: {qpd_epebest}")
            #     # logging.info(f"Current Best Result qpd rmse epoch {qpd_rmseepoch}, result: {qpd_rmsebest}")
            #     # logging.info(f"Current Best Result qpd ai2 epoch {qpd_ai2epoch}, result: {qpd_ai2best}")

            #     # commented temporalily ----------------------------------------------------------------------------------

            #     val_save_skip = 100 // 10
            #     if args.debug_mode:
            #         val_save_skip = 90
            #         # val_save_skip = 100 // 10

            #     results = validate_DPD_Disp(model.module, iters=args.valid_iters, save_result=True, val_save_skip=val_save_skip, datatype=args.datatype, gt_types=['inv_depth'], image_set='test', path='datasets/MDD_dataset', save_path=save_dir)

            #     if dpdisp_ai2best>=results['ai2']:
            #         dpdisp_ai2best = results['ai2']
            #         dpdisp_ai2epoch = epoch
                
            #     logging.info(f"Current Best Result dpdisp ai2 epoch {dpdisp_ai2epoch}, result: {dpdisp_ai2best}")
                
            #     named_results = {}
            #     for k, v in results.items():
            #         named_results[f'val_dpdisp/{k}'] = v
            #         if 'img' not in k:
            #             print(f'val_dpdisp/{k}: {v}')
                
            #     logger.write_dict(named_results)

                model.train()
                # model.module.freeze_bn()

        if total_steps > args.num_steps or (args.stop_step is not None and total_steps > args.stop_step):
            should_keep_training = False
            break

        if len(train_loader) >= 10000:
            model_save_path = os.path.join(args.save_path, 'checkpoints', f'{epoch}_epoch_{total_steps + 1}_{args.name}.pth.gz')
            print()
            logging.info(f"Saving file {model_save_path}")
            torch.save(model.module.state_dict(), model_save_path)
        


    print("FINISHED TRAINING")
    logger.close()
    model_save_path = os.path.join(args.save_path, 'checkpoints', f'final.pth')
    torch.save(model.module.state_dict(), model_save_path)

    return model_save_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='Interp', help="name your experiment")
    parser.add_argument('--restore_ckpt', type=str, default=None, help="restore checkpoint")
    args = parser.parse_args()

    # conf = get_train_config(args.exp_name)
    conf = get_run_setting(args.exp_name)
    
    if args.restore_ckpt:
        conf.restore_ckpt_mono_qpd = args.restore_ckpt
    print(conf)

    train(conf)

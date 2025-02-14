from __future__ import print_function, division

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

from evaluate_mono_qpd import *
import mono_qpd.QPDNet.Quad_datasets as datasets

from argparse import Namespace

from evaluate_mono_qpd import validate_QPD, validate_MDD

from datetime import datetime


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


def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    b,c,h,w = flow_gt.shape
    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any(), "Invalid values in flow predictions"
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)

        fp = flow_preds[i]
        i_loss = (fp-(flow_gt/2)).abs()

        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += i_weight * i_loss[valid.bool()].mean()
    
    fp = flow_preds[-1]
    epe = torch.sum((fp - (flow_gt)/2)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
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


class Logger:

    SUM_FREQ = 100

    def __init__(self, model, scheduler, total_steps, log_dir='result/runs'):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = total_steps
        self.running_loss = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, timestamp))

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.writer = SummaryWriter(log_dir=os.path.join('result/runs', timestamp))

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.writer = SummaryWriter(log_dir=os.path.join('result/runs', timestamp))

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

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

# args.txt 만들기, runs timestamp폴더
def train(args):

    model = MonoQPD(args)
    print("Parameter Count: %d" % count_parameters(model))

    # Codes for debugging NaN
    for name, layer in model.named_modules():
        layer.register_forward_hook(check_nan_hook(name))

    da_v2_args = args['da_v2']
    args = args['else']

    # Prepare the save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_path, timestamp)
    model_save_dir = os.path.join(save_dir, 'checkpoints')
    log_dir = os.path.join(save_dir, 'runs')

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)


    train_loader = datasets.fetch_dataloader(args)
    
    if args.restore_ckpt_mono_qpd is not None:
        assert os.path.exists(args.restore_ckpt_mono_qpd)

        ckpt = torch.load(args.restore_ckpt_mono_qpd)
        total_steps = ckpt['total_steps']
        model.load_state_dict(ckpt['model_state_dict'])

        model = nn.DataParallel(model)
        model.cuda()

        optimizer, scheduler = fetch_optimizer(args, model, -1)
    
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        if not args.initialize_scheduler:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        restore_path_split = args.restore_ckpt_mono_qpd.split('/')

        dst_path = os.path.join(save_dir, restore_path_split[-3]) # [new dir name]/[restored ckpt dir name]
        src_path = '/'.join(restore_path_split[:-2]) # Excluding checkpoints/x.pth [restored ckpt dir name]/
        os.symlink(src_path, dst_path)

        dst_path = os.path.join('/'.join(restore_path_split[:-2]), os.path.basename(save_dir)) # [destination directory name]/[source directory name]
        src_path = save_dir # [new dir name]/
        os.symlink(src_path, dst_path)
    else:
        total_steps = 0
        optimizer, scheduler = fetch_optimizer(args, model, -1)

        if args.restore_ckpt_qpd_net is not None:
            model.qpdnet.load_state_dict(torch.load(args.restore_ckpt_qpd_net))

        if da_v2_args.restore_ckpt_da_v2 is not None:
            model.da_v2.load_state_dict(torch.load(da_v2_args.restore_ckpt_da_v2))

        model = nn.DataParallel(model)
        model.cuda()


    if da_v2_args.freeze_da_v2:
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
        for key, value in vars(da_v2_args).items():
            f.write(f'{key}: {value}\n')

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
        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            center_img, lrtblist, flow, valid = [x.cuda() for x in data_blob]

            assert not torch.isnan(center_img).any(), "Invalid values in input images"
            assert not torch.isnan(lrtblist).any(), "Invalid values in input images"

            b,s,c,h,w = lrtblist.shape

            image1 = center_img.contiguous().view(b,c,h,w)
            if args.input_image_num == 4:
                image2 = torch.cat([lrtblist[:,0],lrtblist[:,1],lrtblist[:,2],lrtblist[:,3]], dim=0).contiguous()
            else:
                image2 = torch.cat([lrtblist[:,0],lrtblist[:,1]], dim=0).contiguous()

            assert model.training
            flow_predictions = model(image1, image2, iters=args.train_iters)
            assert model.training
            if args.input_image_num == 42:
                rot_flow_predictions=[]
                for i in range(len(flow_predictions)):
                    rot_flow_predictions.append(torch.rot90(flow_predictions[i], k=-1, dims=[2,3]))   
                loss, metrics = sequence_loss(rot_flow_predictions, flow, valid)
            elif args.input_image_num == 24:
                rot_flow_predictions=[]
                for i in range(len(flow_predictions)):
                    rot_flow_predictions.append(torch.rot90(flow_predictions[i], k=1, dims=[2,3]))   
                loss, metrics = sequence_loss(rot_flow_predictions, flow, valid)
            else:
                try:
                    loss, metrics = sequence_loss(flow_predictions, flow, valid)
                    
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

            if total_steps % (batch_len*5) == 0  or total_steps==1 or (args.stop_step is not None and total_steps >= args.stop_step):# and total_steps != 0:
                epoch = int(total_steps/batch_len)
                
                model_save_path = os.path.join(args.save_path, timestamp, 'checkpoints', f'{epoch:03d}_epoch_{total_steps}_{args.name}.pth')
                model_save_path = Path(model_save_path).absolute()

                print(os.path.basename(model_save_path))
                logging.info(f"Saving file {model_save_path}")
                torch.save({
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'total_steps': total_steps,
                            # ... any other states you need
                            }, model_save_path)

            if total_steps % (batch_len*5) == 0:
                results = validate_QPD(model.module, iters=args.valid_iters, save_result=False, val_save_skip=30, input_image_num=args.input_image_num, image_set='validation', path='datasets/QP-Data', save_path=save_dir)
                    
                if qpd_epebest>=results['epe']:
                    qpd_epebest = results['epe']
                    qpd_epeepoch = epoch
                if qpd_rmsebest>=results['rmse']:
                    qpd_rmsebest = results['rmse']
                    qpd_rmseepoch = epoch
                if qpd_ai2best>=results['ai2']:
                    qpd_ai2best = results['ai2']
                    qpd_ai2epoch = epoch
                
                
                named_results = {}
                for k, v in results.items():
                    named_results[f'val_qpd/{k}'] = v
                    print(f'val_qpd/{k}: {v}')

                logger.write_dict(named_results)

                logging.info(f"Current Best Result qpd epe epoch {qpd_epeepoch}, result: {qpd_epebest}")
                logging.info(f"Current Best Result qpd rmse epoch {qpd_rmseepoch}, result: {qpd_rmsebest}")
                logging.info(f"Current Best Result qpd ai2 epoch {qpd_ai2epoch}, result: {qpd_ai2best}")

                results = validate_MDD(model.module, iters=args.valid_iters, save_result=False, val_save_skip=30, input_image_num=args.input_image_num, image_set='test', path='datasets/MDD_dataset', save_path=save_dir)

                if dpdisp_ai2best>=results['ai2']:
                    dpdisp_ai2best = results['ai2']
                    dpdisp_ai2epoch = epoch
                
                logging.info(f"Current Best Result dpdisp ai2 epoch {dpdisp_ai2epoch}, result: {dpdisp_ai2best}")
                
                named_results = {}
                for k, v in results.items():
                    named_results[f'val_dpdisp/{k}'] = v
                    print(f'val_dpdisp/{k}: {v}')
                
                logger.write_dict(named_results)

                model.train()
                # model.module.freeze_bn()

        if total_steps > args.num_steps or (args.stop_step is not None and total_steps > args.stop_step):
            should_keep_training = False
            break

        if len(train_loader) >= 10000:
            model_save_path = os.path.join(args.save_path, timestamp, 'checkpoints', f'{epoch}_epoch_{total_steps + 1}_{args.name}.pth.gz')
            print()
            logging.info(f"Saving file {model_save_path}")
            torch.save(model.module.state_dict(), model_save_path)


    print("FINISHED TRAINING")
    logger.close()
    model_save_path = os.path.join(args.save_path, timestamp, 'checkpoints', f'{args.name}.pth')
    torch.save(model.module.state_dict(), model_save_path)

    return model_save_path


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='Mono-QPD', help="name your experiment")
    parser.add_argument('--restore_ckpt_da_v2', default=None, help="restore checkpoint")
    parser.add_argument('--restore_ckpt_qpd_net', default=None, help="restore checkpoint")
    parser.add_argument('--restore_ckpt_mono_qpd', default=None, help="restore checkpoint")

    parser.add_argument('--initialize_scheduler', default=False, action='store_true', help='initialize the scheduler')

    parser.add_argument('--dec_update', default=False, action='store_true', help='initialize the scheduler')

    parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['QPD'], help="training datasets.")
    parser.add_argument('--datasets_path', default='dd_dp_dataset_hypersim_377\\', help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=200000, help="length of training schedule.")
    parser.add_argument('--stop_step', type=int, default=None, help="training stop step(option) ")
    parser.add_argument('--input_image_num', type=int, default=2, help="batch size used during training.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[448, 448], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=8, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")
    parser.add_argument('--CAPA', default=True, help="if use Channel wise and pixel wise attention")
    

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=8, help='number of flow-field updates during validation forward pass')

    # Architecure choices
    parser.add_argument('--corr_implementation', choices=["reg"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")


    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    
    # Depth Anything V2
    parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--img-size', default=518, type=int)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--freeze_da_v2', action='store_true')
    parser.add_argument('--port', default=None, type=int)

    # mono qpd parameters
    parser.add_argument('--feature_converter', type=str, default='', help="training datasets.")
    parser.add_argument('--save_path', type=str, help="path to save")

    args = parser.parse_args()

    # Argument categorization
    da_v2_keys = {'encoder', 'img-size', 'epochs', 'local-rank', 'port', 'restore_ckpt_da_v2', 'freeze_da_v2'}
    else_keys = {'name', 'restore_ckpt_da_v2', 'restore_ckpt_qpd_net', 'restore_ckpt_mono_qpd', 'mixed_precision', 'batch_size', 'train_datasets', 'datasets_path', 'lr', 'num_steps', 'input_image_num', 'image_size', 'train_iters', 'wdecay', 'CAPA', 'valid_iters', 'corr_implementation', 'shared_backbone', 'corr_levels', 'corr_radius', 'n_downsample', 'context_norm', 'slow_fast_gru', 'n_gru_layers', 'hidden_dims', 'img_gamma', 'saturation_range', 'do_flip', 'spatial_scale', 'noyjitter', 'feature_converter', 'save_path', 'stop_step', 'initialize_scheduler', 'dec_update'}

    def split_arguments(args):
        args_dict = vars(args)
        da_v2_args = {key: args_dict[key] for key in da_v2_keys if key in args_dict}
        else_args = {key: args_dict[key] for key in args_dict if key in else_keys}

        return {
            'da_v2': Namespace(**da_v2_args),
            'else': Namespace(**else_args),
        }

    # Split arguments
    split_args = split_arguments(args)

    torch.manual_seed(1234)
    np.random.seed(1234)

    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    # Path("result/checkpoints").mkdir(exist_ok=True, parents=True)
    # Path("result/predictions").mkdir(exist_ok=True, parents=True)
    
    train(split_args)
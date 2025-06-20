from evaluate_mono_qpd import validate_QPD, validate_DPD_Disp, validate_Real_QPD, fix_key, count_parameters
import argparse
from argparse import Namespace
import torch
import torch.nn as nn
import logging
from mono_qpd.mono_qpd import MonoQPD
from glob import glob
import torch.utils.data as data
import os
from exp_args_settings.utils import get_ckpts_in_dir
from exp_args_settings.train_settings import get_train_config
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='Interp', help="name your experiment")
    parser.add_argument('--ckpt_min_epoch', type=int, default=0)
    parser.add_argument('--ckpt_max_epoch', type=int, default=500)
    parser.add_argument('--eval_datasets', choices=['QPD-Test', 'QPD-Valid', 'DPD_Disp', 'Real_QPD'], nargs='+', default=[], required=True, help="Additional dataset to evaluate")

    args = parser.parse_args()

    conf = get_train_config(args.exp_name)

    restore_ckpts = get_ckpts_in_dir(conf.save_path)

    for restore_ckpt in restore_ckpts:
        ckpt = int(os.path.basename(restore_ckpt).split('_')[0])

        if ckpt < args.ckpt_min_epoch or ckpt > args.ckpt_max_epoch:
            continue

        if  ckpt % 5 != 0:
            continue

        model = MonoQPD(conf)
        if restore_ckpt is not None:
            assert restore_ckpt.name.endswith(".pth")
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

        # Delete after model is properly saved
        model = nn.DataParallel(model)

        model.cuda()
        model.eval()

        print("\n")
        print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")
        print(f"Model restored from {restore_ckpt}")

        use_mixed_precision = conf.corr_implementation.endswith("_cuda")
        
        if 'QPD-Test' in args.eval_datasets:
            save_path = os.path.join(conf.save_path, 'qpd-test', os.path.basename(restore_ckpt).replace('.pth', ''))
            print('QPD-Test')
            result = validate_QPD(model, iters=conf.valid_iters, mixed_prec=use_mixed_precision, save_result=False, datatype = conf.datatype, image_set="test", path='datasets/QP-Data', save_path=save_path, batch_size=conf.qpd_test_bs if conf.qpd_test_bs else 1)
        if 'QPD-Valid' in args.eval_datasets:
            save_path = os.path.join(conf.save_path, 'qpd-valid', os.path.basename(restore_ckpt).replace('.pth', ''))
            print('QPD-Valid')
            result = validate_QPD(model, iters=conf.valid_iters, mixed_prec=use_mixed_precision, save_result=False, datatype = conf.datatype, image_set="validation", path='datasets/QP-Data', save_path=save_path, batch_size=conf.qpd_valid_bs if conf.qpd_valid_bs else 1)
        if 'DPD_Disp' in args.eval_datasets:
            save_path = os.path.join(conf.save_path, 'dp-disp', os.path.basename(restore_ckpt).replace('.pth', ''))
            print('DPD_Disp')
            result = validate_DPD_Disp(model, iters=conf.valid_iters, mixed_prec=use_mixed_precision, save_result=False, datatype = conf.datatype, image_set="test", path='datasets/MDD_dataset', save_path=save_path, batch_size=conf.dp_disp_bs if conf.dp_disp_bs else 1)
        if 'Real_QPD' in args.eval_datasets:
            save_path = os.path.join(conf.save_path, 'real-qpd-test', os.path.basename(restore_ckpt).replace('.pth', ''))
            print('Real_QPD')
            result = validate_Real_QPD(model, iters=conf.valid_iters, mixed_prec=use_mixed_precision, save_result=False, datatype = conf.datatype, image_set="test", path='datasets/Real-QP-Data', save_path=save_path, batch_size=conf.real_qpd_bs if conf.real_qpd_bs else 1)

        print(result)

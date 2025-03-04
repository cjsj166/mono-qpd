from evaluate_mono_qpd import validate_QPD, validate_MDD, validate_Real_QPD, fix_key, count_parameters
import argparse
from argparse import Namespace
import torch
import torch.nn as nn
import logging
from mono_qpd.mono_qpd import MonoQPD
from glob import glob
import torch.utils.data as data
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', help="Path to the file containing the model names to evaluate", type=str) # restore ckpts
    
    parser.add_argument('--datasets', nargs='+', help="dataset for evaluation", default=["QPD-Test"])
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
    parser.add_argument('--save_result', type=bool, default='True')
    parser.add_argument('--save_name', default='val')
    
    # Evaluation settings
    parser.add_argument('--save_path', default='result/validations/eval.txt')
    parser.add_argument('--ckpt_min_epoch', type=int, default=0)
    parser.add_argument('--ckpt_max_epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=4, help="batch size for evaluation")

    # Depth Anything V2
    parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--feature_converter', type=str, default='', help="training datasets.")

    
    args = parser.parse_args()

    # args.save_result = args.save_result == str(True)

    # Argument categorization
    da_v2_keys = {'encoder', 'img-size', 'epochs', 'local-rank', 'port', 'restore_ckpt_da_v2', 'freeze_da_v2'}
    else_keys = {'name', 'restore_ckpt_da_v2', 'restore_ckpt_qpd_net', 'mixed_precision', 'batch_size', 'datasets', 'datasets_path', 'lr', 'num_steps', 'input_image_num', 'image_size', 'train_iters', 'wdecay', 'CAPA', 'valid_iters', 'corr_implementation', 'shared_backbone', 'corr_levels', 'corr_radius', 'n_downsample', 'context_norm', 'slow_fast_gru', 'n_gru_layers', 'hidden_dims', 'img_gamma', 'saturation_range', 'do_flip', 'spatial_scale', 'noyjitter', 'feature_converter', 'save_path'}

    def split_arguments(args):
        args_dict = vars(args)
        da_v2_args = {key: args_dict[key] for key in da_v2_keys if key in args_dict}
        else_args = {key: args_dict[key] for key in args_dict if key in else_keys}

        return {
            'da_v2': Namespace(**da_v2_args),
            'else': Namespace(**else_args),
        }
    
    split_args = split_arguments(args)
    

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    restore_ckpts = glob(os.path.join(args.train_dir, '**','checkpoints', '*.pth'))

    for restore_ckpt in restore_ckpts:
        ckpt = int(os.path.basename(restore_ckpt).split('_')[0])

        if ckpt < args.ckpt_min_epoch or ckpt > args.ckpt_max_epoch:
            continue

        if  ckpt % 5 != 0:
            continue

        model = MonoQPD(split_args)
        if restore_ckpt is not None:
            assert restore_ckpt.endswith(".pth")
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

        print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")

        use_mixed_precision = args.corr_implementation.endswith("_cuda")
        
        if 'QPD-Test' in args.datasets:
            save_path = os.path.join(args.save_path, 'qpd-test', os.path.basename(restore_ckpt).replace('.pth', ''))
            print(save_path)
            result = validate_QPD(model, iters=args.valid_iters, mixed_prec=use_mixed_precision, save_result=False, input_image_num = args.input_image_num, image_set="test", path='datasets/QP-Data', save_path=save_path)
        if 'QPD-Valid' in args.datasets:
            save_path = os.path.join(args.save_path, 'qpd-valid', os.path.basename(restore_ckpt).replace('.pth', ''))
            print(save_path)
            result = validate_QPD(model, iters=args.valid_iters, mixed_prec=use_mixed_precision, save_result=False, input_image_num = args.input_image_num, image_set="validation", path='datasets/QP-Data', save_path=save_path)
        if 'MDD' in args.datasets:
            save_path = os.path.join(args.save_path, 'dp-disp', os.path.basename(restore_ckpt).replace('.pth', ''))
            print(save_path)
            result = validate_MDD(model, iters=args.valid_iters, mixed_prec=use_mixed_precision, save_result=False, input_image_num = args.input_image_num, image_set="test", path='datasets/MDD_dataset', save_path=save_path)
        if 'Real_QPD' in args.datasets:
            save_path = os.path.join(args.save_path, 'real-qpd-test', os.path.basename(restore_ckpt).replace('.pth', ''))
            print(save_path)
            result = validate_Real_QPD(model, iters=args.valid_iters, mixed_prec=use_mixed_precision, save_result=False, input_image_num = args.input_image_num, image_set="test", path=args.datasets_path, save_path=save_path)

        print(result)

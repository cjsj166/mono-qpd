import argparse
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from pathlib import Path
# import train_mono_qpd

@dataclass
class TrainConfig:
    name: str = 'Mono-QPD'
    restore_ckpt_da_v2: str = None
    restore_ckpt_qpd_net: str = None
    restore_ckpt_mono_qpd: str = None
    initialize_scheduler: bool = False
    dec_update: bool = False
    mixed_precision: bool = False
    
    # Training parameters
    batch_size: int = 4
    train_datasets: Tuple[str] = ('QPD',)
    datasets_path: str = 'datasets/QP-Data'
    lr: float = 0.0002
    num_steps: int = 600000
    stop_step: int = None
    input_image_num: int = 2
    image_size: Tuple[int] = (224, 224)
    train_iters: int = 8
    wdecay: float = 0.00001
    CAPA: bool = True
    si_loss: float = 0.0

    # Validation parameters
    valid_iters: int = 8

    # Architecure choices
    corr_implementation: str = 'reg'
    shared_backbone: bool = False
    corr_levels: int = 4
    corr_radius: int = 4
    n_downsample: int = 2
    context_norm: str = 'batch'
    slow_fast_gru: bool = False
    n_gru_layers: int = 3
    hidden_dims: Tuple[int] = (128, 128, 128)

    # Data augmentation
    img_gamma: Tuple[float] = None
    saturation_range: Tuple[float] = None
    do_flip: str = None
    spatial_scale: Tuple[float] = (0, 0)
    noyjitter: bool = False
    datatype: str = 'dual'
    qpd_gt_types: Tuple[str] = ('disp',)
    dp_disp_gt_types: Tuple[str] = ('inv_depth',)

    # Depth Anything V2
    encoder: str = 'vitl'
    img_size: int = 518
    epochs: int = 40
    local_rank: int = 0
    freeze_da_v2: bool = True
    port: int = None
    feature_converter: str = ''
    save_path: str = None
    qpd_valid_bs: int = 1
    val_save_skip: int = 50

    # evaluation settings
    qpd_valid_bs: int = 1
    qpd_test_bs: int = 1
    real_qpd_bs: int = 1
    dp_disp_bs: int = 1
    val_datasets: Tuple[str] = ('QPD-Valid',)
    eval_datasets: Tuple[str] = ('QPD-Test', 'Real-QPD', 'DPD-Disp')

    def __post_init__(self):
        save_path = Path('result/train') / self.__class__.__name__

        self.save_path = str(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='interp', help="name your experiment")
    args = parser.parse_args()

    conf_dict = get_train_config(args.exp_name)
    conf_namespace = parser.Namespace(**conf_dict)
    print(conf_namespace)

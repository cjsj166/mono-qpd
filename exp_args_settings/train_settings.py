import argparse
from dataclasses import dataclass
from typing import List
import train_mono_qpd

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
    train_datasets: List[str] = ('QPD',)
    datasets_path: str = 'dd_dp_dataset_hypersim_377\\'
    lr: float = 0.0002
    num_steps: int = 200000
    stop_step: int = None
    input_image_num: int = 2
    image_size: List[int] = (448, 448)
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
    hidden_dims: List[int] = (128, 128, 128)

    # Data augmentation
    img_gamma: List[float] = None
    saturation_range: List[float] = None
    do_flip: str = None
    spatial_scale: List[float] = (0, 0)
    noyjitter: bool = False
    datatype: str = 'dual'
    qpd_gt_types: List[str] = ('disp',)
    dp_disp_gt_types: List[str] = ('inv_depth',)

    # Depth Anything V2
    encoder: str = 'vitl'
    img_size: int = 518
    epochs: int = 40
    local_rank: int = 0
    freeze_da_v2: bool = False
    port: int = None
    feature_converter: str = ''
    save_path: str = None

    @classmethod
    def interp(cls):
        return cls(
            batch_size=1,
            restore_ckpt_da_v2='mono_qpd/Depth_Anything_V2/checkpoints/depth_anything_v2_vitl.pth',
            image_size=(224, 224),
            datasets_path='datasets/QP-Data',
            freeze_da_v2=True,
            save_path='result/train/',
            feature_converter='interp'
            )
    
    @classmethod
    def tsubame_interp(cls):
        conf = cls.interp()
        conf.batch_size = 16
        conf.restore_ckpt_da_v2 = ''
        conf.image_size = (448, 448)
        conf.datasets_path = 'datasets/QP-Data'
        conf.freeze_da_v2 = True
        conf.save_path = 'result/train/'
        conf.feature_converter = 'interp'

        return conf

def get_train_config(exp_name):
    return getattr(TrainConfig, exp_name)()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='interp', help="name your experiment")
    args = parser.parse_args()

    conf_dict = get_train_config(args.exp_name)
    conf_namespace = parser.Namespace(**conf_dict)
    print(conf_namespace)

    train_mono_qpd.train(conf_namespace)

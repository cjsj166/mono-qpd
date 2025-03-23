from dataclasses import dataclass
from .train_settings import TrainConfig
# from train_settings import TrainConfig # For debugging
from typing import Tuple
import numpy as np

@dataclass
class Interp(TrainConfig):
    num_steps: int = 200000
    # FIXME: batch_size -> 12
    batch_size: int = 1
    image_size: Tuple[int, int] = (448, 448)
    lr=np.sqrt(8) * 0.0002
    qpd_valid_bs: int = 4
    qpd_test_bs: int = 4
    real_qpd_bs: int = 4
    dp_disp_bs: int = 4
    val_save_skip: int = 1

    restore_ckpt_da_v2: str = 'mono_qpd/Depth_Anything_V2/checkpoints/depth_anything_v2_vitl.pth'
    save_path: str = 'result/train/Interp'
    feature_converter: str = 'interp'
    val_datasets: Tuple[str] = ('DPD-Disp',)

@dataclass
class Interp200K(TrainConfig):
    num_steps: int = 200000
    batch_size: int = 12
    image_size: Tuple[int, int] = (448, 448)
    lr=np.sqrt(12) * 0.0002
    qpd_valid_bs: int = 4
    qpd_test_bs: int = 4
    real_qpd_bs: int = 4
    dp_disp_bs: int = 4
    val_save_skip: int = 1

    restore_ckpt_da_v2: str = 'mono_qpd/Depth_Anything_V2/checkpoints/depth_anything_v2_vitl.pth'
    save_path: str = 'result/train/200k_Interp'
    feature_converter: str = 'interp'
    val_datasets: Tuple[str] = ('DPD-Disp',)

@dataclass
class InterpOriginal(TrainConfig):
    num_steps: int = 600000
    batch_size: int = 2
    image_size: Tuple[int, int] = (448, 448)
    lr: int = 0.0002
    qpd_valid_bs: int = 4
    qpd_test_bs: int = 4
    real_qpd_bs: int = 4
    dp_disp_bs: int = 4
    val_save_skip: int = 1

    restore_ckpt_da_v2: str = 'mono_qpd/Depth_Anything_V2/checkpoints/depth_anything_v2_vitl.pth'
    save_path: str = 'result/train/InterpOriginal'
    feature_converter: str = 'interp'
    val_datasets: Tuple[str] = ('DPD-Disp',)

@dataclass
class InterpQPDSetting(TrainConfig):
    num_steps: int = 200000
    batch_size: int = 4
    image_size: Tuple[int, int] = (448, 448)
    lr: int = 0.0002
    qpd_valid_bs: int = 4
    qpd_test_bs: int = 4
    real_qpd_bs: int = 4
    dp_disp_bs: int = 4
    val_save_skip: int = 1

    restore_ckpt_da_v2: str = 'mono_qpd/Depth_Anything_V2/checkpoints/depth_anything_v2_vitl.pth'
    save_path: str = 'result/train/InterpQPDSetting'
    feature_converter: str = 'interp'
    val_datasets: Tuple[str] = ('DPD-Disp',)

@dataclass
class Interp150K(TrainConfig):
    num_steps: int = 150000
    batch_size: int = 16
    image_size: Tuple[int, int] = (448, 448)
    lr: int = 0.0002 * np.sqrt(16)
    qpd_valid_bs: int = 4
    qpd_test_bs: int = 4
    real_qpd_bs: int = 4
    dp_disp_bs: int = 4
    val_save_skip: int = 1

    restore_ckpt_da_v2: str = 'mono_qpd/Depth_Anything_V2/checkpoints/depth_anything_v2_vitl.pth'
    save_path: str = 'result/train/Interp150K'
    feature_converter: str = 'interp'
    val_datasets: Tuple[str] = ('DPD-Disp',)

@dataclass
class Interp75K(TrainConfig):
    num_steps: int = 75_000
    batch_size: int = 16
    image_size: Tuple[int, int] = (448, 448)
    lr: int = 0.0002 * np.sqrt(16)
    qpd_valid_bs: int = 4
    qpd_test_bs: int = 4
    real_qpd_bs: int = 4
    dp_disp_bs: int = 4
    val_save_skip: int = 1

    restore_ckpt_da_v2: str = 'mono_qpd/Depth_Anything_V2/checkpoints/depth_anything_v2_vitl.pth'
    save_path: str = 'result/train/Interp75K'
    feature_converter: str = 'interp'
    val_datasets: Tuple[str] = ('DPD-Disp',)

if __name__ == "__main__":
    conf = Interp()
    print(conf)


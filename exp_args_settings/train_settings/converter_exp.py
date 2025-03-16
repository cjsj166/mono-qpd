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
    lr=np.sqrt(12) * 0.0002
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


if __name__ == "__main__":
    conf = Interp()
    print(conf)
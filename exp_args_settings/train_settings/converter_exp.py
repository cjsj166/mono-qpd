from dataclasses import dataclass
from .train_settings import TrainConfig
# from train_settings import TrainConfig # For debugging
from typing import Tuple

@dataclass
class Interp(TrainConfig):
    batch_size: int = 2
    restore_ckpt_da_v2: str = 'mono_qpd/Depth_Anything_V2/checkpoints/depth_anything_v2_vitl.pth'
    image_size: Tuple[int, int] = (224, 224)
    datasets_path: str = 'datasets/QP-Data'
    freeze_da_v2: bool = True
    save_path: str = 'result/train/Interp'
    feature_converter: str = 'interp'

if __name__ == "__main__":
    conf = Interp()
    print(conf)
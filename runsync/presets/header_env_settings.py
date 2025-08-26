from dataclasses import dataclass
from dataclasses import asdict

@dataclass
class BaseConfig:
    node_type: str = "gpu_h"
    running_time: str = "24:00:00"
    env_name: str = "env"
    cuda_version: str = "cuda/12.1"
    cudnn_version: str = "cudnn/9.0.0"

# --- user input ---

@dataclass
class FMDPTrain(BaseConfig):
    """Header env setting for FMDP jobs"""
    node_type: str = "gpu_1"   # 요청하신 job script와 동일
    running_time: str = "24:00:00"
    env_name: str = "mono-qpd"  # conda activate mono-qpd
    cuda_version: str = "cuda/12.1"
    cudnn_version: str = "cudnn/9.0.0"

@dataclass
class FMDPValid(BaseConfig):
    """Header env setting for FMDP jobs"""
    node_type: str = "gpu_1"   # 요청하신 job script와 동일
    running_time: str = "00:15:00"
    env_name: str = "mono-qpd"  # conda activate mono-qpd
    cuda_version: str = "cuda/12.1"
    cudnn_version: str = "cudnn/9.0.0"

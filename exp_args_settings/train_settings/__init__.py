from .converter_exp import Interp
from .train_settings import *

def get_train_config(class_name: str):
    if class_name in globals():
        return globals()[class_name]()
    else:
        raise ValueError(f"Class {class_name} not found")
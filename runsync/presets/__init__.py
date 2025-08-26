from . import run_settings
from . import header_env_settings

def get_run_setting(class_name: str):
    if hasattr(run_settings, class_name):
        cls = getattr(run_settings, class_name)
        return cls()
    else:
        raise ValueError(f"Class '{class_name}' not found in pre-defined settings.")

def get_header_env_setting(class_name: str):
    if hasattr(header_env_settings, class_name):
        cls = getattr(header_env_settings, class_name)
        return cls()
    else:
        raise ValueError(f"Class '{class_name}' not found in pre-defined settings.")

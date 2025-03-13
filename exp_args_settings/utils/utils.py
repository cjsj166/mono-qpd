from pathlib import Path


def extract_epoch(ckpt):
    ckpt = Path(ckpt)
    return int(str(ckpt.name).split("_")[0])

def get_latest_ckpt(path):
    pth = Path(path)
    ckpts = pth.glob("**/*.pth")

    sorted_ckpts = sorted(ckpts, key=extract_epoch)

    if len(sorted_ckpts) == 0:
        return None
    
    latest_ckpt = sorted_ckpts[-1]

    return latest_ckpt

def get_ckpts_in_dir(dir_path):
    ckpts_pattern = Path(dir_path) / '**' / '*.pth'
    ckpts = ckpts_pattern.rglob()
    ckpts = sorted(ckpts, key=extract_epoch)
    return ckpts 

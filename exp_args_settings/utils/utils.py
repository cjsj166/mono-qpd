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
    dir_path = Path(dir_path)
    ckpts = dir_path.rglob('**/*.pth')
    ckpts = sorted(ckpts, key=extract_epoch)
    return ckpts 

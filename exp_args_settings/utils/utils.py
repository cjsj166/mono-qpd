from pathlib import Path


def extract_epoch(ckpt):
    ckpt = Path(ckpt)
    try :
        ckpt_num = int(str(ckpt.name).split("_")[0])
        return ckpt_num
    except ValueError as e:
        print(f'{e}')
        return None

def get_latest_ckpt(path):
    pth = Path(path)
    ckpts = pth.glob("**/*.pth")

    valid_ckpts = [ckpt for ckpt in ckpts if extract_epoch(ckpt)]
    
    sorted_ckpts = sorted(valid_ckpts, key=extract_epoch)

    if len(sorted_ckpts) == 0:
        return None
    
    latest_ckpt = sorted_ckpts[-1]

    return latest_ckpt

def get_ckpts_in_dir(dir_path):
    dir_path = Path(dir_path)
    ckpts = dir_path.rglob('**/*.pth')
    pat = os.path.join(dir_path, '*', 'checkpoints', '*.pth')
    ckpts = [Path(ckpt) for ckpt in glob(pat)]

    valid_ckpts = [ckpt for ckpt in ckpts if extract_epoch(ckpt)]

    ckpts = sorted(valid_ckpts, key=extract_epoch)
    return ckpts 

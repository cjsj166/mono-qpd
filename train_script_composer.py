import os
import argparse
from exp_args_settings.train_settings import get_train_config
from pathlib import Path
from datetime import datetime

def extract_epoch(ckpt):
    return int(str(ckpt.name).split("_")[0])

def get_latest_ckpt(path):
    pth = Path(path)
    ckpts = pth.glob("**/*.pth")

    sorted_ckpts = sorted(ckpts, key=extract_epoch)

    assert len(sorted_ckpts) > 0, "No checkpoint found"
    
    latest_ckpt = sorted_ckpts[-1]

    return latest_ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_script', default='scripts/basic_script.sh', help="name your experiment")
    parser.add_argument('--exp_name', default='Interp', help="name your experiment")
    parser.add_argument('--tsubame', action='store_true', help="when you run on tsubame")
    parser.add_argument('-r', action='store_true', help="resume train until it reaches the max epoch")
    
    args = parser.parse_args()

    script_path = Path(args.base_script)

    script = []
    with script_path.open() as f:
        script = f.readlines()

    # change the job name as [the experiment name + current time stamp]
    script[2] = f"#$ -N train_{args.exp_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}\n"

    script[-1] = f"python train.py --exp_name {args.exp_name}"

    # get the train config
    if args.tsubame:
        dcl = get_train_config(args.exp_name)
        conf = dcl.tsubame()
        script[-1] = script[-1] + f" --tsubame"
    else:
        conf = get_train_config(args.exp_name)

    # get the latest checkpoint
    if args.r:
        latest_ckpt = get_latest_ckpt(conf.save_path)
        script[-1] = script[-1] + f" --restore_ckpt {latest_ckpt}"

    total_epoch = conf.num_steps / 3010

    script = "".join(script)    
    print(script)



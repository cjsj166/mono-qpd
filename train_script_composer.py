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

    if len(sorted_ckpts) == 0:
        return None
    
    latest_ckpt = sorted_ckpts[-1]

    return latest_ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_script', default='scripts/basic_script.sh', help="name your experiment")
    parser.add_argument('--exp_name', default='Interp', required=True, help="name your experiment")
    parser.add_argument('--tsubame', action='store_true', help="when you run on tsubame")
    parser.add_argument('--restore_ckpt', default=None, help="restore the checkpoint")
    
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

    # get the latest checkpoint if user don't specify the checkpoint
    if args.restore_ckpt:
        load_ckpt = Path(args.restore_ckpt)
        script[-1] = script[-1] + f" --restore_ckpt {args.restore_ckpt}"
        load_ckpt_epoch = extract_epoch(load_ckpt)
    else:
        latest_ckpt = get_latest_ckpt(conf.save_path)
        if latest_ckpt:
            load_ckpt = latest_ckpt
            script[-1] = script[-1] + f" --restore_ckpt {latest_ckpt}"
            load_ckpt_epoch = extract_epoch(load_ckpt)
        else:
            load_ckpt_epoch = 0

    # if total_epoch is already reached, do not train
    if 'QPD' in conf.train_datasets:
        total_epoch = conf.num_steps / 3010
        if load_ckpt_epoch >= total_epoch:
            print("Already reached the max epoch")
            exit(0)

    script = "".join(script)

    # save the script file
    save_path = Path("scripts") / args.exp_name
    save_path.mkdir(parents=True, exist_ok=True)

    if load_ckpt:
        save_path = save_path / "resume_train.sh"
    else:
        save_path = save_path / "train.sh"

    with save_path.open("w") as f:
        f.write(script)
    
    print(f"cd {save_path.parent}; sub {save_path}")



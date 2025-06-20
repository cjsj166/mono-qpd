import os
import argparse
from exp_args_settings.train_settings import get_train_config
from pathlib import Path
from datetime import datetime
from exp_args_settings.utils import extract_epoch, get_latest_ckpt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_script', default='scripts/basic_script.sh', help="name your experiment")
    parser.add_argument('--exp_name', default='Interp', required=True, help="name your experiment")
    parser.add_argument('--restore_ckpt', default=None, help="restore the checkpoint")
    
    args = parser.parse_args()

    script_path = Path(args.base_script)

    script = []
    with script_path.open() as f:
        script = f.readlines()

    # change the job name as [the experiment name + current time stamp]
    script[2] = f"#$ -N train_{args.exp_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}\n"

    script[-1] = f"python train_mono_qpd.py --exp_name {args.exp_name}"

    # get the train config
    conf = get_train_config(args.exp_name)

    # get the latest checkpoint if user don't specify the checkpoint
    load_ckpt = None
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
        total_epoch = int(conf.num_steps * conf.batch_size / 3010)
        if load_ckpt_epoch >= total_epoch - total_epoch % 5:
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
    
    print(f"cd {save_path.parent};sub {save_path.name}")



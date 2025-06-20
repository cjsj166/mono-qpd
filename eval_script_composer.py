import os
import argparse
from pathlib import Path
from datetime import datetime
from exp_args_settings.train_settings import get_train_config
from exp_args_settings.utils import get_ckpts_in_dir, extract_epoch
import math

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True, help="Experiment name")
    parser.add_argument('--ckpt_epoch', type=int, default=None, help="Specific epoch to evaluate")
    parser.add_argument('--ckpt_min_epoch', type=int, default=0)
    parser.add_argument('--ckpt_max_epoch', type=int, default=500)
    parser.add_argument('--job_num', type=int, default=1, help="Number of parallel jobs")
    parser.add_argument('--eval_datasets', choices=['QPD-Test', 'QPD-Test-noise', 'QPD-Valid', 'DPD_Disp', 'Real_QPD'], nargs='+', default=[], required=True, help="Additional dataset to evaluate")
    
    args = parser.parse_args()

    script_path = Path('scripts') / 'basic_script.sh'
    script = []
    with script_path.open() as f:
        script = f.readlines()

    # change the job name as [the experiment name + current time stamp]
    script[3] = f"#$ -l gpu_h=1\n"
    script[4] = f"#$ -l h_rt=6:00:00\n"
    script[2] = f"#$ -N multi_eval_{args.exp_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}\n"
    script[-1] = f"python evaluate_multiple_models.py --exp_name {args.exp_name}"

    # get the train config
    conf = get_train_config(args.exp_name)

    save_dir = Path(f"scripts/{args.exp_name}/evaluations")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # get the checkpoints from train directory
    ckpts = get_ckpts_in_dir(conf.save_path)

    if not ckpts:
        print("No checkpoint found")
        exit(0)
    
    if args.ckpt_epoch:
        script_path = save_dir / f"{args.ckpt_epoch:03d}.sh"
        script[-1] = script[-1].replace("evaluate_multiple_models.py", "evaluate_mono_qpd.py")
        script[-1] = script[-1] + f" --ckpt_epoch {args.ckpt_epoch}"
        script[-1] = script[-1] + f" --eval_datasets {' '.join(args.eval_datasets)}\n"

        with script_path.open("w") as f:
            script = "".join(script)
            f.write(script)
        
        print(f"cd {script_path.parent};sub {script_path.name}")
    else:
        min_epoch, max_epoch = extract_epoch(ckpts[0]), extract_epoch(ckpts[-1])
        job_size = max(1, math.ceil(len(ckpts) / args.job_num))
        job_scripts = []
        
        for i in range(0, len(ckpts), job_size):
            job_ckpts = ckpts[i:i + job_size]
            job_min_epoch, job_max_epoch = extract_epoch(job_ckpts[0]), extract_epoch(job_ckpts[-1])
            script_path = save_dir / f"{job_min_epoch}_{job_max_epoch}.sh"
            job_scripts.append(script_path)
            
            job_script = script[:]
            job_script[-1] = job_script[-1] + f" --ckpt_min_epoch {job_min_epoch} --ckpt_max_epoch {job_max_epoch} --eval_datasets {' '.join(args.eval_datasets)}"
            with script_path.open("w") as f:
                job_script = "".join(job_script)
                f.write(job_script)
        
        print(";".join([f"cd {save_dir}; sub {job_script_path.name}" for job_script_path in job_scripts]))

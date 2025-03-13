#!/bin/bash
#$ -cwd
#$ -N multi_eval_Interp_20250313170532
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -V
. /etc/profile.d/modules.sh
source  ~/.bashrc
module load cuda/12.1 cudnn/9.0.0
conda activate mono-qpd
cd /gs/bs/tga-lab_otm/dlee/mono-qpd
python evaluate_multiple_models.py --exp_name Interp --ckpt_min_epoch 25 --ckpt_max_epoch 40
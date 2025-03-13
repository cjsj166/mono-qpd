#!/bin/bash
#$ -cwd
#$ -N eval_interp-eval
#$ -l gpu_h=1
#$ -l h_rt=24:00:00
#$ -V
. /etc/profile.d/modules.sh
source  ~/.bashrc
module load cuda/12.1 cudnn/9.0.0
nvidia-smi
conda activate mono-qpd
cd /gs/bs/tga-lab_otm/dlee/mono-qpd
python evaluate_multiple_models.py \
	--train_dir result/train/exp_interp/ \
	--datasets QPD-Valid \
	--save_path result/eval/exp_interp/ \
	--feature_converter interp \
	--qpd_valid_bs 8 \
	--ckpt_min_epoch 0 \
	--ckpt_max_epoch 50

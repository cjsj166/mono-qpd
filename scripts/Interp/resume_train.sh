#!/bin/bash
#$ -cwd
#$ -N train_Interp_20250312014916
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -V
. /etc/profile.d/modules.sh
source  ~/.bashrc
module load cuda/12.1 cudnn/9.0.0
conda activate mono-qpd
cd /gs/bs/tga-lab_otm/dlee/mono-qpd
python train_mono_qpd.py --exp_name Interp --tsubame --restore_ckpt result/train/Interp/20250311_222842/checkpoints/025_epoch_9400_Mono-QPD.pth
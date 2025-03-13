#!/bin/bash
#$ -cwd
#$ -N train_interp
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -V
. /etc/profile.d/modules.sh
source  ~/.bashrc
module load cuda/12.1 cudnn/9.0.0
nvidia-smi
conda activate mono-qpd
cd /gs/bs/tga-lab_otm/dlee/mono-qpd
record_nvidia_smi gpu_log.log
python train_mono_qpd.py --exp_name Interp --tsubame

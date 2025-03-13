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


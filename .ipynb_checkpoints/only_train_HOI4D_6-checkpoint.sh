#!/bin/bash
#SBATCH  --time=4:00:00
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gpus=rtx_2080_ti:1
#SBATCH  -A ls_hilli
#SBATCH  -n 4
#SBATCH  --mem-per-cpu=10000
workon Def3DGS

DATA_DIR=/cluster/project/hilliges/mbressieux/data/HOI4D/HOI4D_video6_nerfied
LOG=/cluster/project/hilliges/mbressieux/log/HOI4D/HOI4D_video6_nerfied

python train.py -s ${DATA_DIR} -m ${LOG} --eval --iterations 20000

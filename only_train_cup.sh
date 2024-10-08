#!/bin/bash
#SBATCH  --time=4:00:00
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gpus=rtx_3090:1
#SBATCH  -A ls_hilli
#SBATCH  -n 4
#SBATCH  --mem-per-cpu=10000
workon Def3DGS

DATA_DIR=/cluster/project/hilliges/mbressieux/data/NeRF/cup_novel_view
LOG=/cluster/project/hilliges/mbressieux/log/cup_novel_view_low_l1_high_thresh

python train.py -s ${DATA_DIR} -m ${LOG} --eval --iterations 20000

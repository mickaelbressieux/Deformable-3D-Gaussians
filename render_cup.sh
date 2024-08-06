#!/bin/bash
#SBATCH  --time=4:00:00
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gpus=rtx_2080_ti:1
#SBATCH  -A ls_hilli
#SBATCH  -n 4
#SBATCH  --mem-per-cpu=10000

LOG=/cluster/project/hilliges/mbressieux/log/cup_novel_view_new

python render.py -m ${LOG} --mode segment --eval

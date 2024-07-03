#!/bin/bash
#SBATCH  --time=48:00:00
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gpus=rtx_2080_ti:1
#SBATCH  -A ls_hilli
#SBATCH  -n 4
#SBATCH  --mem-per-cpu=15000
workon Def3DGS

DATA_DIR=/cluster/project/hilliges/mbressieux/data/cu_tom_nerfied
LOG=/cluster/project/hilliges/mbressieux/log/cu_tom_nerfied

python train.py -s ${DATA_DIR} -m ${LOG} --eval --iterations 20000

#!/bin/bash
#SBATCH  --time=4:00:00
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gpus=rtx_3090:1
#SBATCH  -A ls_hilli
#SBATCH  -n 4
#SBATCH  --mem-per-cpu=10000
workon Def3DGS

DATA_DIR=/cluster/project/hilliges/mbressieux/data/EK_Capstone/NeRF/P03_03_Clip_nerfied
LOG=/cluster/project/hilliges/mbressieux/log/EK_Capstone/P03_03_Clip_nerfied_capstone

python train.py -s ${DATA_DIR} -m ${LOG} --eval --iterations 20000

#!/bin/bash

DATA_DIR=/M2SSD1/Datasets/NeRF-DS.dataset/sieve_novel_view
LOG=/M2SSD1/Logs/Def_3DGS/sieve_novel_view

python train.py -s ${DATA_DIR} -m ${LOG} --eval --iterations 20000

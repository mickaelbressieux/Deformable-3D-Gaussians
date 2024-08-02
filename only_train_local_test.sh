#!/bin/bash

DATA_DIR=/M2SSD1/Datasets/NeRF-DS.dataset/cup_novel_view
LOG=/M2SSD1/Logs/Def_3DGS/cup_novel_view_test

python train.py -s ${DATA_DIR} -m ${LOG} --eval --iterations 20000

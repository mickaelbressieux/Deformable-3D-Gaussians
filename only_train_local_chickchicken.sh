#!/bin/bash

DATA_DIR=/M2SSD1/Datasets/chickchicken
LOG=/M2SSD1/Logs/Def_3DGS/chickchicken

python train.py -s ${DATA_DIR} -m ${LOG} --eval --iterations 20000

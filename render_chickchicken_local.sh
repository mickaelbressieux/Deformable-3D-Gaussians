#!/bin/bash

LOG=/M2SSD1/Logs/Def_3DGS/chickchicken

python render.py -m ${LOG} --mode segment --skip_test

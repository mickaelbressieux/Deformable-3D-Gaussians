#!/bin/bash

LOG=/M2SSD1/Logs/Def_3DGS/cup_novel_view

python render.py -m ${LOG} --mode segment --skip_test

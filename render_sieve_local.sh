#!/bin/bash

LOG=/M2SSD1/Logs/Def_3DGS/sieve_novel_view

python render.py -m ${LOG} --mode oneCamera --skip_test

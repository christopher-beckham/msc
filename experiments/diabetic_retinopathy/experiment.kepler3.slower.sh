#!/bin/bash

LOW_RES_N2_BASELINE_CROP_RESUME_QWKCF_BS512_BUT_LOWER_LR=1 \
THEANO_FLAGS=mode=FAST_RUN,device=gpu1,lib.cnmem=0.95,allow_gc=True,floatX=float32,nvcc.fastmath=True,profile=False \
  python -u exp_trick_experiments.py

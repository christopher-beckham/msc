#!/bin/bash

LOW_RES_N2_BASELINE_CROP_QWK_BALANCED_BS512_S1_LR005=1 \
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,lib.cnmem=0.95,allow_gc=True,floatX=float32,nvcc.fastmath=True,profile=False \
  python -u exp_trick_experiments.py

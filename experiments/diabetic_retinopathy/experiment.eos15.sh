#!/bin/bash

LOW_RES_N2_BASELINE_CROP_HACKY_ORDINAL_S2=1 \
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,lib.cnmem=0.95,allow_gc=True,floatX=float32,nvcc.fastmath=True,profile=False,dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once \
  python -u exp_trick_experiments.py

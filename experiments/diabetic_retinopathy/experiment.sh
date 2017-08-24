#!/bin/bash

LOW_RES_N2_BASELINE_CROP_QWKREFORM_LEARNEND_S1=1 \
THEANO_FLAGS=mode=FAST_RUN,device=gpu,lib.cnmem=0.95,allow_gc=True,floatX=float32,nvcc.fastmath=True,profile=False,dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once \
  python -u exp_trick_experiments.py

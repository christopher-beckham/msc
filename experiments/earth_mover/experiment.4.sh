#!/usr/bin/env bash

env
EARTH_MOVER=`pwd` \
CUDA_LAUNCH_BLOCKING=0 \
THEANO_FLAGS=mode=FAST_RUN,profile=False,device=cuda,lib.cnmem=0.95,allow_gc=True,floatX=float32,nvcc.fastmath=True,dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once \
  python experiments.py dr_sq_backrelu_fixed_l2_1e4_sgd_pre_split_hdf5_adam train 1

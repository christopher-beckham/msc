#!/bin/bash

CUDA_LAUNCH_BLOCKING=0 \
THEANO_FLAGS=mode=FAST_RUN,lib.cnmem=0.95,device=gpu1,allow_gc=True,floatX=float32,nvcc.fastmath=True,profile=False,dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once \
  python -u vis_conv_net_gpu.py

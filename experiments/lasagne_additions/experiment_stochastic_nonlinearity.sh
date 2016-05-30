#!/bin/bash

CUDA_LAUNCH_BLOCKING=0 \
CIFAR10_EXP_2=1 \
THEANO_FLAGS=mode=FAST_RUN,device=gpu1,allow_gc=True,floatX=float32,nvcc.fastmath=True,profile=False,dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once \
  python -u stochastic_depth_resnet.py

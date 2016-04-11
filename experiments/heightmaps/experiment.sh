#!/bin/bash

NAME=vgg_a_subset_beefed
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,nvcc.fastmath=True,dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once \
  python -u experiment.py output/${NAME}.pkl > output/${NAME}.txt

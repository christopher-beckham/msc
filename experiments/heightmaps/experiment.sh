#!/bin/bash

NAME=vgg_a_subset_usevalid
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,nvcc.fastmath=True \
  python -u experiment.py output/${NAME}.pkl > output/${NAME}.txt
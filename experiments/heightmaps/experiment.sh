#!/bin/bash

NAME=only_one_example.2
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,nvcc.fastmath=True \
  python -u experiment.py output/${NAME}.pkl > output/${NAME}.txt

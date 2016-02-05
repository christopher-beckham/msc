#!/bin/bash

THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,dnn.conv.algo_fwd=time_once python vis_transformer.py

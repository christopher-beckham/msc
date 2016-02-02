#!/bin/bash

THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,dnn.conv.algo_fwd=time_once python -u run_exp.py

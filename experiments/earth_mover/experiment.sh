#!/bin/bash

# experiments.py <name of experiment> <mode> <seed>
# note: i removed fastmath for testing purposes
THEANO_FLAGS=mode=FAST_RUN,device=cuda,lib.cnmem=0.95,allow_gc=True,floatX=float32,profile=False,dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once \
  python experiments.py dr_xent_l2_1e4_sgd_pre_split_hdf5_adam dump_dist_test 1

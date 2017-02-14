#!/usr/bin/env bash

env
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export THEANO_FLAGS=mode=FAST_RUN,profile=False,device=cuda,lib.cnmem=0.95,allow_gc=True,floatX=float32,nvcc.fastmath=True,dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME
export EARTH_MOVER=`pwd`
python experiments.py dr_pois_t_5_xent_l2_1e4_sgd_pre_split_hdf5

#!/bin/bash

# useful? https://groups.google.com/forum/#!topic/theano-users/vHY0h5Gdu6w

THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float64 \
python ../style_transfer.py \
    --config_name=../configurations/vgg19_cpu.py \
    --model_name=../vgg19_normalized.pkl \
    --npy_file=../train_data_minimal.npy \
    --ref_image_index=5 \
    --style_coef=1e7 \
    --variation_coef=0.001 \
    --num_images=1 \
    --num_iters=6 \
    --out_folder=../output_vgg

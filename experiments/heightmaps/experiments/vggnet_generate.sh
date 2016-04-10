#!/bin/bash

# useful? https://groups.google.com/forum/#!topic/theano-users/vHY0h5Gdu6w

# 1460255930_evolution.png was generated with vgg19_cpu.py, with style_coef=1e7,variation_coef=0.001
# 1460257664_evolution.png was generated with style_coef=1e7,variation_coef=0.01

THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float64 \
python ../style_transfer.py \
    --config_name=../configurations/vgg19_cpu.py \
    --model_name=../vgg19_normalized.pkl \
    --npy_file=../train_data_minimal.npy \
    --ref_image_index=5 \
    --style_coef=1e8 \
    --variation_coef=0.001 \
    --num_images=1 \
    --num_iters=6 \
    --out_folder=../output_vgg

#!/bin/bash

# vgg_a_subset_usevalid.pkl

# 1460421579_evolution.png with var=0.0001
python ../style_transfer.py \
    --config_name=../configurations/vgg_a.py \
    --model_name=../output/vgg_a.pkl \
    --npy_file=../train_data_minimal.npy \
    --ref_image_index=5 \
    --style_coef=1e7 \
    --variation_coef=0.0001 \
    --num_images=1 \
    --num_iters=15 \
    --grid=no \
    --sigma=0 \
    --out_folder=../output_neat_new

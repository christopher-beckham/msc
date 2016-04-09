#!/bin/bash

python ../style_transfer.py \
    --config_name=../configurations/vgg_a_subset.py \
    --model_name=../output/vgg_a_subset.2.pkl \
    --npy_file=../train_data_minimal.npy \
    --ref_image_index=5 \
    --style_coef=1e7 \
    --variation_coef=0.001 \
    --num_images=1 \
    --num_iters=6 \
    --out_folder=../output_neat
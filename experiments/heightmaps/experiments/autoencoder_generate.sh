#!/bin/bash

# vgg_a_subset_usevalid.pkl

# 1460421579_evolution.png with var=0.0001

# 0.001, 0.0001, 0.00001

#for var in 0.000001 0.0000001 0.00000001; do
python ../style_transfer.py \
    --config_name=../configurations/vgg_a_subset_less_depth.py \
    --model_name=../output/vgg_a_subset_less_depth.pkl \
    --npy_file=../train_data_minimal.npy \
    --ref_image_index=5 \
    --style_coef=1e7 \
    --variation_coef=0.0001 \
    --num_images=1 \
    --num_iters=6 \
    --grid=no \
    --sigma=8 \
    --out_folder=../output_neat_new
#done

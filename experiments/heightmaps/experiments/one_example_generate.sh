#!/bin/bash

# vgg_a_subset_usevalid.pkl

# 1460421579_evolution.png with var=0.0001

# 0.001, 0.0001, 0.00001

for style in 1e7 1e8 1e9 1e10; do
#for var in 0.000001 0.0000001 0.00000001; do
python ../style_transfer.py \
    --config_name=../configurations/vgg_a_subset_less_depth.py \
    --model_name=../output/only_one_example.pkl \
    --npy_file=../train_data_minimal.npy \
    --ref_image_index=5 \
    --style_coef=$style \
    --variation_coef=0 \
    --num_images=1 \
    --num_iters=6 \
    --grid=yes \
    --outfile=../output_neat_new/var0_${style}
#done
done

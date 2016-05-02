#!/bin/bash




python ../style_transfer.py \
    --config_name=../configurations/vgg_a_subset_less_depth.py \
    --model_name=../output/only_one_example.pkl \
    --npy_file=../train_data_minimal.npy \
    --ref_image_index=5 \
    --style_coef=1e7 \
    --variation_coef=1e-5 \
    --num_images=1 \
    --num_iters=6 \
    --grid=yes \
    --cheat_index=4 \
    --outfile=../output_neat_new/cheat

#!/bin/bash

# Single test run makes a x2 slow-motion
CUDA_VISIBLE_DEVICES=1 python main.py \
    --dataset test \
    --data_root demo \
    --img_fmt jpg \
    --exp_name superslomo \
    --model superslomo \
    --test_batch_size 1 \
    --loss 1*Super \
    --optimizer Adam \
    --inner_lr 1e-5 \
    --outer_lr 1e-5 \
    --number_of_evaluation_steps_per_iter 1 \
    --mode test \
    --pretrained_model pretrained_models/meta_superslomo.pth 

# Repeating the same command again will make x4 slow-motion
CUDA_VISIBLE_DEVICES=1 python main.py \
    --dataset test \
    --data_root demo \
    --img_fmt jpg \
    --exp_name superslomo \
    --model superslomo \
    --test_batch_size 1 \
    --loss 1*Super \
    --optimizer Adam \
    --inner_lr 1e-5 \
    --outer_lr 1e-5 \
    --number_of_evaluation_steps_per_iter 1 \
    --mode test \
    --pretrained_model pretrained_models/meta_superslomo.pth 

# Repeating the same commands more will make x2^N slow-motion by interpolating repeatedly

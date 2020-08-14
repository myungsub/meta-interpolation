#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python main.py \
    --exp_name dain-metasgd \
    --model dain \
    --loss 1*L1 \
    --optimizer Adamax \
    --batch_size 6 \
    --val_batch_size 1 \
    --inner_lr 1e-5 \
    --outer_lr 1e-5 \
    --total_iter_per_epoch 3000 \
    --number_of_training_steps_per_iter 1 \
    --number_of_evaluation_steps_per_iter 1 \
    --log_iter 10 \
    --mode val \
    --dataset hd \
    --data_root data/HD_dataset/HD_RGB \
    --resume \
    --metasgd
#    --pretrained_model pretrained_models/meta_dain_candidate2.pth \
#    --pretrained_model pretrained_models/dain_base.pth \
#    --learnable_per_layer_per_step_inner_loop_learning_rate \
#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python main.py \
    --exp_name voxelflow-metasgd-i1 \
    --model voxelflow \
    --batch_size 8 \
    --val_batch_size 1 \
    --loss 1*MSE \
    --optimizer Adam \
    --inner_lr 1e-5 \
    --outer_lr 1e-5 \
    --total_iter_per_epoch 5000 \
    --number_of_training_steps_per_iter 1 \
    --number_of_evaluation_steps_per_iter 1 \
    --log_iter 10 \
    --metasgd
#    --mode val \
#    --pretrained_model pretrained_models/meta_voxelflow.pth
#    --resume \
#    --learnable_per_layer_per_step_inner_loop_learning_rate
#    --second_order
#    --first_order_to_second_order_epoch 100
#    --use_multi_step_loss_optimization
#    --multi_step_loss_num_epochs 1
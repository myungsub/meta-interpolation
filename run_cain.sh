#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python main.py \
    --exp_name cain-metasgd-i1 \
    --model cain \
    --loss 1*L1 \
    --optimizer Adam \
    --batch_size 8 \
    --val_batch_size 1 \
    --inner_lr 1e-5 \
    --outer_lr 1e-5 \
    --total_iter_per_epoch 5000 \
    --number_of_training_steps_per_iter 1 \
    --number_of_evaluation_steps_per_iter 1 \
    --log_iter 10 \
    --num_workers 9 \
    --metasgd
#    --mode val
#    --learnable_per_layer_per_step_inner_loop_learning_rate
#    --second_order
#    --first_order_to_second_order_epoch 100
#    --use_multi_step_loss_optimization
#    --multi_step_loss_num_epochs 1
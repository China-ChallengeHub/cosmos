#!/usr/bin/env bash

# eg: 实验组
# ck: 对照组
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_roberta_adam.py \
    --do_train \
    --do_lower_case \
    --learning_rate=1e-5 \
    --max_seq_length=256 \
    --model_choice=base \
    --mask_type=commonsense \
    --num_train_epochs=5 \
    --weight_decay=0.01 \
    --adam_epsilon=1e-8 \
    --logging_steps=200 \
    --save_steps=200 \
    --output_dir=../../checkpoint/single/output_base_lr_1e-5_bz_16_epoch_5_adam_commonsense0_256

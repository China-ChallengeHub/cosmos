#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_roberta_adam.py \
    --do_train \
    --do_lower_case \
    --model_choice=base \
    --learning_rate=2e-5 \
    --num_train_epochs=5 \
    --max_seq_length=256 \
    --weight_decay=0.01 \
    --adam_epsilon=1e-6 \
    --logging_steps=200 \
    --save_steps=200 \
    --output_dir=../../checkpoint/baseline/roberta/output_base_lr_2e-5_bz_16_epoch_5_adam_256

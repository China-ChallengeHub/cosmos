#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_roberta_adamw.py \
    --do_train \
    --do_lower_case \
    --model_choice=base \
    --mask_type=commonsense \
    --learning_rate=2e-5 \
    --max_seq_length=256 \
    --warmup_steps=0 \
    --num_train_epochs=5 \
    --weight_decay=0.01 \
    --adam_epsilon=1e-6 \
    --logging_steps=200 \
    --save_steps=200 \
    --output_dir=../../checkpoint/single/test

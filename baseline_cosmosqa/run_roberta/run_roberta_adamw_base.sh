#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_roberta_adamw.py \
    --do_train \
    --do_lower_case \
    --model_choice=base \
    --learning_rate=2e-5 \
    --num_train_epochs=5 \
    --max_seq_length=256 \
    --weight_decay=0.01 \
    --adam_epsilon=1e-8 \
    --logging_steps=200 \
    --save_steps=200 \
    --output_dir=../../checkpoint/baseline/output_base_lr_2e-5_bz_16_epoch_5_warmup_rate_0.1_adamw_256

#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_roberta_adamw.py \
    --do_train \
    --do_lower_case \
    --model_choice=base \
    --bert_model_choice=fix_head \
    --max_seq_length=256 \
    --learning_rate=2e-5 \
    --num_train_epochs=5 \
    --weight_decay=0.01 \
    --adam_epsilon=1e-6 \
    --logging_steps=200 \
    --save_steps=200 \
    --output_dir=../../checkpoint/fusion/output_base_lr_2e-5_bz_16_epoch_5_adamw_warmup_rate_0.1_fix_head

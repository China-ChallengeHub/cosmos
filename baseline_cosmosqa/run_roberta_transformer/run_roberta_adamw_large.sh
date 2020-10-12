#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_roberta_adamw.py \
    --do_train \
    --do_lower_case \
    --model_choice=large \
    --learning_rate=1e-5 \
    --max_seq_length=180 \
    --num_train_epochs=5 \
    --weight_decay=0.01 \
    --adam_epsilon=1e-6 \
    --logging_steps=500 \
    --save_steps=500 \
    --output_dir=../../output_model/output_cosmosqa/output_roberta_transformer/output_large_lr_1e-5_bz_4_epoch_5_adamw_warmup_rate_0.1

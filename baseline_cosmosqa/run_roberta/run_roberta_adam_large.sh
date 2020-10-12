#!/usr/bin/env bash

# eg: 实验组
# ck: 对照组
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_roberta_adam.py \
    --do_train \
    --do_lower_case \
    --learning_rate=5e-6 \
    --model_choice=large \
    --num_train_epochs=10 \
    --weight_decay=0.01 \
    --adam_epsilon=1e-6 \
    --logging_steps=500 \
    --save_steps=500 \
    --output_dir=../../output_model/output_cosmosqa/output_roberta/output_large_lr_5e-6_bz_4_adam

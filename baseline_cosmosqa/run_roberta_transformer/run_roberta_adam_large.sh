#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_roberta_adam.py \
    --do_train \
    --do_lower_case \
    --model_choice=large \
    --learning_rate=5e-6 \
    --num_train_epochs=5 \
    --weight_decay=0.01 \
    --adam_epsilon=1e-6 \
    --logging_steps=200 \
    --save_steps=200 \
    --output_dir=../../output_model/output_cosmosqa/output_roberta_transformer/output_large_lr_5e-6_bz_8_epoch_5_adam

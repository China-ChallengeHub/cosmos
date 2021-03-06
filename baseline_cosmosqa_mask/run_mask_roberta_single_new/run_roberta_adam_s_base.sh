#!/usr/bin/env bash

# eg: 实验组
# ck: 对照组
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_roberta_adam.py \
    --do_train \
    --do_lower_case \
    --learning_rate=2e-5 \
    --max_seq_length=256 \
    --model_choice=base \
    --mask_type=sentiment \
    --num_train_epochs=5 \
    --weight_decay=0.01 \
    --adam_epsilon=1e-6 \
    --logging_steps=200 \
    --save_steps=200 \
    --output_dir=../../output_model/output_cosmosqa_mask/output_roberta_transformer_new/output_base_lr_2e-5_bz_16_epoch_5_adam_sentiment

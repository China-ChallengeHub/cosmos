#!/usr/bin/env bash

# eg: 实验组
# ck: 对照组
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_roberta_v0_adamw.py \
    --do_train \
    --do_lower_case \
    --learning_rate=2e-5 \
    --model_choice=base \
    --mask_type=commonsense \
    --num_train_epochs=5 \
    --weight_decay=0.01 \
    --adam_epsilon=1e-6 \
    --logging_steps=200 \
    --save_steps=200 \
    --output_dir=../../../output_model/output_cosmosqa_mask/output_roberta_transformer/output_base_lr_2e-5_bz_16_epoch_5_adamw_warmup_rate_0.1_commonsense

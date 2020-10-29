#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_roberta_adamw_single.py \
    --do_train \
    --do_lower_case \
    --model_choice=base \
    --bert_model_choice=fusion_layer \
    --max_seq_length=256 \
    --logging_steps=200 \
    --save_steps=200 \
    --output_dir=../../checkpoint/fusion/attn/weight_decay_0.01/adam_epsilon_1e-6/output_base_lr_1e-5_bz_12_epoch_5_adamw_warmup_step_0_fusion_layer_256_sentiment
#    --output_dir=../../checkpoint/fusion/attn/weight_decay_0.01/adam_epsilon_1e-6/output_base_lr_1e-5_bz_12_epoch_5_adamw_warmup_step_0_fusion_layer_256_commonsense
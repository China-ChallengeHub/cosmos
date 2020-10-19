#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_roberta_adamw.py \
    --do_train \
    --do_lower_case \
    --model_choice=base \
    --bert_model_choice=fusion_layer \
    --max_seq_length=256 \
    --logging_steps=200 \
    --save_steps=200 \
    --output_dir=../../checkpoint/fusion/single_attn/weight_decay_0.01/adam_epsilon_1e-8/output_base_lr_1e-5_bz_12_epoch_5_adamw_warmup_step_0_commonsense_256
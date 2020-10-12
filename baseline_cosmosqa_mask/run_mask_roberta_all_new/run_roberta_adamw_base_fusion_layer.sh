#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_roberta_adamw.py \
    --do_train \
    --do_lower_case \
    --model_choice=base \
    --bert_model_choice=fusion_layer \
    --max_seq_length=256 \
    --warmup_step=1000 \
    --learning_rate=1e-5 \
    --num_train_epochs=5 \
    --weight_decay=0.01 \
    --adam_epsilon=1e-6 \
    --logging_steps=200 \
    --save_steps=200 \
    --output_dir=../../checkpoint/fusion/new/output_base_lr_1e-5_bz_12_epoch_5_adamw_warmup_step_1000_fusion_layer

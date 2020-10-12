#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_roberta_adamw.py \
    --do_train \
    --do_lower_case \
    --model_choice=base \
    --bert_model_choice=fusion_head \
    --max_seq_length=256 \
    --num_train_epochs=5 \
    --logging_steps=200 \
    --save_steps=200 \
    --output_dir=../../checkpoint/fusion/attn/output_base_lr_1e-5_bz_16_epoch_5_adamw_warmup_step_0_fusion_layer

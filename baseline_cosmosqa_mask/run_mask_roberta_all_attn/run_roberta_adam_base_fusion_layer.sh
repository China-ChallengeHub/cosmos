#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_roberta_adam.py \
    --do_train \
    --do_lower_case \
    --model_choice=base \
    --bert_model_choice=fusion_layer \
    --learning_rate=1e-5 \
    --max_seq_length=256 \
    --logging_steps=200 \
    --save_steps=200 \
    --output_dir=../../checkpoint/fusion/attn/output_base_lr_1e-5_bz_12_epoch_5_adam_fusion_layer_256_nobyte
#    --output_dir=../../checkpoint/fusion/attn/output_base_lr_1e-5_bz_12_epoch_5_adam_fusion_layer_256
#    --output_dir=../../checkpoint/fusion/attn/output_base_lr_5e-6_bz_12_epoch_5_adam_fusion_layer_256

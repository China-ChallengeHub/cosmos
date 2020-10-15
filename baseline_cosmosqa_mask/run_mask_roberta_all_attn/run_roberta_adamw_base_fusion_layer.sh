#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_roberta_adamw.py \
    --do_train \
    --do_lower_case \
    --model_choice=base \
    --bert_model_choice=fusion_layer \
    --max_seq_length=256 \
    --logging_steps=200 \
    --save_steps=200 \
    --output_dir=../../checkpoint/fusion/attn/adam_epsilon/output_base_lr_1e-5_bz_32_epoch_5_adamw_warmup_step_0_fusion_layer_256
#    --output_dir=../../checkpoint/fusion/attn/weight_decay/adam_epsilon_1e-8/output_base_lr_2e-5_bz_16_epoch_5_adamw_warmup_step_0_fusion_layer_256
#    --output_dir=../../checkpoint/fusion/attn/weight_decay/adam_epsilon_1e-8/output_base_lr_1e-5_bz_16_epoch_5_adamw_warmup_step_0_fusion_layer_256
#    --output_dir=../../checkpoint/fusion/attn/weight_decay/adam_epsilon_1e-8/output_base_lr_1e-5_bz_12_epoch_5_adamw_warmup_step_0_fusion_layer_256
#    --output_dir=../../checkpoint/fusion/attn/weight_decay/adam_epsilon_1e-6/output_base_lr_1e-5_bz_12_epoch_5_adamw_warmup_step_0_fusion_layer_256
#    --output_dir=../../checkpoint/fusion/attn/adam_epsilon/output_base_lr_1e-5_bz_12_epoch_5_adamw_warmup_rate_0.1_fusion_layer_256
#    --output_dir=../../checkpoint/fusion/attn/adam_epsilon/output_base_lr_1e-5_bz_16_epoch_5_adamw_warmup_step_0_fusion_layer_256
#    --output_dir=../../checkpoint/fusion/attn/adam_epsilon/output_base_lr_1e-5_bz_12_epoch_5_adamw_warmup_step_0_fusion_layer_256
#    --output_dir=../../checkpoint/fusion/attn/output_base_lr_2e-5_bz_12_epoch_5_adamw_warmup_rate_0.1_fusion_layer_256_bp
#    --output_dir=../../checkpoint/fusion/attn/output_base_lr_2e-5_bz_12_epoch_5_adamw_warmup_step_1500_fusion_layer_256
#    --output_dir=../../checkpoint/fusion/attn/output_base_lr_2e-5_bz_12_epoch_5_adamw_warmup_step_1000_fusion_layer_256
#    --output_dir=../../checkpoint/fusion/attn/output_base_lr_2e-5_bz_12_epoch_5_adamw_warmup_step_0_fusion_layer_256
#    --output_dir=../../checkpoint/fusion/attn/output_base_lr_2e-5_bz_16_epoch_5_adamw_warmup_step_1500_fusion_layer_256
#    --output_dir=../../checkpoint/fusion/attn/output_base_lr_2e-5_bz_16_epoch_5_adamw_warmup_step_1000_fusion_layer_256
#    --output_dir=../../checkpoint/fusion/attn/output_base_lr_2e-5_bz_16_epoch_5_adamw_warmup_step_0_fusion_layer_256
#    --output_dir=../../checkpoint/fusion/attn/output_base_lr_1e-5_bz_12_epoch_5_adamw_warmup_step_1500_fusion_layer_256
#    --output_dir=../../checkpoint/fusion/attn/output_base_lr_1e-5_bz_12_epoch_5_adamw_warmup_step_1000_fusion_layer_256
#    --output_dir=../../checkpoint/fusion/attn/output_base_lr_1e-5_bz_12_epoch_5_adamw_warmup_step_0_fusion_layer_256
#    --output_dir=../../checkpoint/fusion/attn/output_base_lr_1e-5_bz_16_epoch_5_adamw_warmup_step_1500_fusion_layer_256
#    --output_dir=../../checkpoint/fusion/attn/output_base_lr_1e-5_bz_16_epoch_5_adamw_warmup_step_1000_fusion_layer_256
#    --output_dir=../../checkpoint/fusion/attn/output_base_lr_1e-5_bz_16_epoch_5_adamw_warmup_step_0_fusion_layer_256

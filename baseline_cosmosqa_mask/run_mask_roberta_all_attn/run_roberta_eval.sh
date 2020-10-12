#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_roberta_eval.py \
    --do_eval \
    --do_lower_case \
    --model_choice=base \
    --bert_model_choice=fusion_layer \
    --learning_rate=1e-5 \
    --warmup_step=0 \
    --max_seq_length=256 \
    --num_train_epochs=5 \
    --weight_decay=0.01 \
    --adam_epsilon=1e-6 \
    --logging_steps=200 \
    --save_steps=200 \
    --output_dir=../../checkpoint/fusion/attn/output_base_lr_1e-5_bz_12_epoch_5_adamw_warmup_step_0_fusion_layer

#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_roberta_adamw_predict.py \
    --do_train \
    --do_lower_case \
    --model_choice=base \
    --bert_model_choice=fusion_layer \
    --learning_rate=2e-5 \
    --max_seq_length=256 \
    --num_train_epochs=5 \
    --weight_decay=0.01 \
    --adam_epsilon=1e-6 \
    --logging_steps=200 \
    --save_steps=200 \
    --output_dir=../../output_model/output_cosmosqa_mask/output_roberta_transformer_ensemble_attn/output_base_lr_2e-5_bz_12_epoch_5_adamw_predict_warmup_step_0_fusion_layer

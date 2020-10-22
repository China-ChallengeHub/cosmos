#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_roberta_eval.py \
    --do_eval \
    --do_lower_case \
    --model_choice=base \
    --bert_model_choice=fusion_layer \
    --max_seq_length=256 \

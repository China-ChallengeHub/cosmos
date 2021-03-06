from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import torch
import logging

cur_path = os.path.abspath(__file__)
cur_dir = os.path.dirname(cur_path)
par_dir = os.path.dirname(cur_dir)
gra_dir = os.path.dirname(par_dir)
grg_dir = os.path.dirname(gra_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)
sys.path.append(gra_dir)
sys.path.append(grg_dir)

from pytorch_pretrained_xbert import RobertaConfig
from pytorch_pretrained_xbert import RobertaTokenizer

from baseline_cosmosqa_mask.run_mask_roberta_single_attn.roberta_adam import train
from baseline_cosmosqa_mask.run_mask_roberta_single_attn.roberta_adam import set_seed
from baseline_cosmosqa_mask.run_mask_roberta_single_attn.config import parse_args
from baseline_cosmosqa_mask.run_mask_roberta_single_attn.util_feature import read_features

from baseline_cosmosqa_mask.run_mask_roberta_all_attn.util_feature import InputFeatures
from baseline_cosmosqa_mask.run_mask_roberta_all_attn.util_feature import CommonsenseFeatures
from baseline_cosmosqa_mask.run_mask_roberta_all_attn.util_feature import DependencyFeatures
from baseline_cosmosqa_mask.run_mask_roberta_all_attn.util_feature import EntityFeatures
from baseline_cosmosqa_mask.run_mask_roberta_all_attn.util_feature import SentimentFeatures

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)

    if args.model_choice == "large":
        args.per_gpu_train_batch_size = 1
        args.per_gpu_eval_batch_size = 2
        args.model_name_or_path = os.path.join(grg_dir, "pretrained_model/roberta-large")
    elif args.model_choice == "base":
        args.per_gpu_train_batch_size = 3
        args.per_gpu_eval_batch_size = 4
        args.model_name_or_path = os.path.join(grg_dir, "pretrained_model/roberta-base")
        print("args.per_gpu_train_batch_size = ", args.per_gpu_train_batch_size)
    else:
        raise ValueError

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size  = args.per_gpu_eval_batch_size  * max(1, args.n_gpu)
    print("args.train_batch_size = ", args.train_batch_size)
    print("args.eval_batch_size = ", args.eval_batch_size)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args_path = os.path.join(args.output_dir, "args.json")
    with open(args_path, "w", encoding="utf-8") as writer:
        json.dump(args.__dict__, writer, ensure_ascii=False, indent=4)

    if args.model_choice == "base":
        if args.bert_model_choice == "fusion_head":
            from baseline_cosmosqa_mask.model.model_mask_roberta_all.model_base_attn import \
                RobertaForMultipleChoice_Fusion_Head as RobertaForMultipleChoice
        elif args.bert_model_choice == "fusion_layer":
            from baseline_cosmosqa_mask.model.model_mask_roberta_all.model_base_attn import \
                RobertaForMultipleChoice_Fusion_Layer as RobertaForMultipleChoice
        elif args.bert_model_choice == "fusion_head_bert_attn":
            from baseline_cosmosqa_mask.model.model_mask_roberta_all.model_base_attn import \
                RobertaForMultipleChoice_Fusion_Head_Bert_Self_Attn as RobertaForMultipleChoice
        else:
            raise ValueError
    elif args.model_choice == "large":
        if args.bert_model_choice == "fusion_head":
            from baseline_cosmosqa_mask.model.model_mask_roberta_all.model_large_attn import \
                RobertaForMultipleChoice_Fusion_Head as RobertaForMultipleChoice
        elif args.bert_model_choice == "fusion_layer":
            from baseline_cosmosqa_mask.model.model_mask_roberta_all.model_large_attn import \
                RobertaForMultipleChoice_Fusion_Layer as RobertaForMultipleChoice
        elif args.bert_model_choice == "fusion_head_bert_attn":
            from baseline_cosmosqa_mask.model.model_mask_roberta_all.model_large_attn import \
                RobertaForMultipleChoice_Fusion_Head_Bert_Self_Attn as RobertaForMultipleChoice
        else:
            raise ValueError
    else:
        raise ValueError

    config_class, model_class, tokenizer_class = RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=4, finetuning_task=args.task_name)

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)

    # model = model_class.from_pretrained(args.model_name_or_path,
    #                                     from_tf=bool('.ckpt' in args.model_name_or_path),
    #                                     config=config)
    model = model_class.from_pretrained(args.model_name_or_path)

    train_dataset, dev_dataset = read_features(args)

    # Training
    if args.do_train:
        global_step, tr_loss, best_step = train(args, train_dataset, dev_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s, best_step = %s", global_step, tr_loss, best_step)


if __name__ == "__main__":
    main()

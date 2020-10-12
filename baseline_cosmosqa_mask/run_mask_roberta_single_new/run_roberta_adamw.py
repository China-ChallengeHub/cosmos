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

from baseline_cosmosqa_mask.run_mask_roberta_single_new.config import parse_args
from baseline_cosmosqa_mask.run_mask_roberta_single_new.roberta_adamw import train
from baseline_cosmosqa_mask.run_mask_roberta_single_new.roberta_adamw import set_seed

from baseline_cosmosqa_mask.run_mask_roberta_single_new.util_feature import InputFeatures
from baseline_cosmosqa_mask.run_mask_roberta_single_new.util_feature import CommonsenseFeatures
from baseline_cosmosqa_mask.run_mask_roberta_single_new.util_feature import DependencyFeatures
from baseline_cosmosqa_mask.run_mask_roberta_single_new.util_feature import EntityFeatures
from baseline_cosmosqa_mask.run_mask_roberta_single_new.util_feature import SentimentFeatures

from baseline_cosmosqa_mask.run_mask_roberta_single_new.util_feature import read_commonsense_features
from baseline_cosmosqa_mask.run_mask_roberta_single_new.util_feature import read_dependency_features
from baseline_cosmosqa_mask.run_mask_roberta_single_new.util_feature import read_entity_features
from baseline_cosmosqa_mask.run_mask_roberta_single_new.util_feature import read_sentiment_features

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
        args.per_gpu_train_batch_size = 4
        args.per_gpu_eval_batch_size = 4
        args.model_name_or_path = os.path.join(grg_dir, "pretrained_model/roberta-base")
    else:
        raise ValueError

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size  = args.per_gpu_eval_batch_size  * max(1, args.n_gpu)
    print("args.train_batch_size = ", args.train_batch_size)
    print("args.eval_batch_size = ", args.eval_batch_size)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.model_choice == "base":
        from baseline_cosmosqa_mask.model.model_mask_roberta_single.model_base import RobertaForMultipleChoice
    elif args.model_choice == "large":
        from baseline_cosmosqa_mask.model.model_mask_roberta_single.model_large import RobertaForMultipleChoice
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

    if args.mask_type == "commonsense":
        train_dataset, dev_dataset = read_commonsense_features(args)
    elif args.mask_type == "dependency":  # dependency mask 默认为从子节点到所有祖宗节点
        train_dataset, dev_dataset = read_dependency_features(args)
    elif args.mask_type == "entity":
        train_dataset, dev_dataset = read_entity_features(args)
    elif args.mask_type == "sentiment":
        train_dataset, dev_dataset = read_sentiment_features(args)
    else:
        raise ValueError

    # Training
    if args.do_train:
        global_step, tr_loss, best_step = train(args, train_dataset, dev_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s, best_step = %s", global_step, tr_loss, best_step)


if __name__ == "__main__":
    main()

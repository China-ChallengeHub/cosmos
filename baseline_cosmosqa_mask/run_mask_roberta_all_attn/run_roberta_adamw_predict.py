from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import time
import torch
import pandas
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

from baseline_cosmosqa_mask.run_mask_roberta_all_attn.config import parse_args
from baseline_cosmosqa_mask.run_mask_roberta_all_attn.roberta_adamw import eval
from baseline_cosmosqa_mask.run_mask_roberta_all_attn.roberta_adamw import predict
from baseline_cosmosqa_mask.run_mask_roberta_all_attn.roberta_adamw import set_seed
from baseline_cosmosqa_mask.run_mask_roberta_all_attn.util_feature_test import read_features_test
from baseline_cosmosqa_mask.run_mask_roberta_all_attn.util_feature import read_features

from baseline_cosmosqa_mask.run_mask_roberta_all_attn.util_feature_test import InputFeatures
from baseline_cosmosqa_mask.run_mask_roberta_all_attn.util_feature_test import CommonsenseFeatures
from baseline_cosmosqa_mask.run_mask_roberta_all_attn.util_feature_test import DependencyFeatures
from baseline_cosmosqa_mask.run_mask_roberta_all_attn.util_feature_test import EntityFeatures
from baseline_cosmosqa_mask.run_mask_roberta_all_attn.util_feature_test import SentimentFeatures

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
        args.model_name_or_path = "../../pretrained_model/_roberta-large"
    elif args.model_choice == "base":
        args.per_gpu_train_batch_size = 3
        args.per_gpu_eval_batch_size = 4
        args.model_name_or_path = "../../pretrained_model/_roberta-base"
    else:
        raise ValueError

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size  = args.per_gpu_eval_batch_size  * max(1, args.n_gpu)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

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

    model_dir = "../../output_model/output_cosmosqa_mask/output_roberta_transformer_ensemble_attn/" \
        "output_base_lr_1e-5_bz_12_epoch_5_adamw_warmup_step_0_fusion_layer/"

    # model = model_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(model_dir)

    model.cuda()
    if args.n_gpu > 1:
        gpu_ids = list(range(args.n_gpu))
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.eval()

    test_dataset = read_features_test(args)
    # train_dataset, dev_dataset = read_features(args)

    # Training
    if args.do_train:
        print("[TIME] --- time: {} ---, start predict".format(time.ctime(time.time())))
        # results = eval(args, model, dev_dataset, prefix="", test=False)
        # print("eval results = ", results)

        preds = predict(args, model, test_dataset, prefix="", test=False)
        df = pandas.DataFrame(preds, columns=["one"])
        df.to_csv("./prediction.lst", columns=["one"], index=False, header=False)


if __name__ == "__main__":
    main()

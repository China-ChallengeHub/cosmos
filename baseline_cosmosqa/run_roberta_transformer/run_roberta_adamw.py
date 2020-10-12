from __future__ import absolute_import, division, print_function

import os
import sys
import json
import time
import torch
import pickle
import random
import logging
import argparse
import numpy as np

from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from torch.utils.data import TensorDataset
from torch.utils.data.distributed import DistributedSampler

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

from pytorch_pretrained_xbert import AdamW
from pytorch_pretrained_xbert import WarmupLinearSchedule

from baseline_cosmosqa.run_roberta_transformer.utils_cosmosqa import CosmosProcessor
from baseline_cosmosqa.run_roberta_transformer.visual_util import update_lr
from baseline_cosmosqa.run_roberta_transformer.visual_util import update_epoch_loss
from baseline_cosmosqa.run_roberta_transformer.visual_util import update_step_result
from baseline_cosmosqa.run_roberta_transformer.visual_util import update_eval_accuracy

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def init_optimizer(args, train_dataloader, model):
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.warmup_steps = int(t_total * 0.1)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    model.cuda()
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        gpu_ids = list(range(args.n_gpu))
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    return model, optimizer, scheduler, t_total


def train(args, train_dataset, dev_dataset, model, tokenizer):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    model, optimizer, scheduler, t_total = init_optimizer(args, train_dataloader, model)

    # # Train!
    # logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", len(train_dataset))
    # logger.info("  Num Epochs = %d", args.num_train_epochs)
    # logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    # logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
    #             args.train_batch_size * args.gradient_accumulation_steps * (
    #             torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    # logger.info("  Total optimization steps = %d", t_total)

    tr_step, global_step = 0, 0
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_acc, best_dev_loss = 0.0, 99999999999.0
    best_step = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for epoch in train_iterator:
        epoch_step, epoch_loss = 0, 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()

            args.device = torch.device("cuda")
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            tr_step += 1

            epoch_loss += loss.item()
            epoch_step += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

                lr_this_step = scheduler.get_lr()[0]
                update_lr(lr_this_step, global_step)

                assert args.logging_steps == args.save_steps
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    results = eval(args, model, dev_dataset, prefix="", test=False)

                    # TODO add visdom visualization
                    eval_loss = results["eval_loss"]
                    eval_accu = results["eval_acc"]
                    update_step_result(eval_loss, eval_accu, global_step)

                    if eval_accu > best_dev_acc:
                        best_dev_acc = eval_accu
                        best_dev_loss = eval_loss
                        best_step = global_step

                        model_to_save = model.module if hasattr(model, 'module') else model
                        model_to_save.save_pretrained(args.output_dir)
                        model_path = os.path.join(args.output_dir, f'training_args.bin')
                        torch.save(args, model_path)
                        result_path = os.path.join(args.output_dir, f"eval_result_{global_step}.json")
                        with open(result_path, "w", encoding="utf-8") as writer:
                            json.dump(results, writer, ensure_ascii=False, indent=4)

                    train_loss = tr_loss - logging_loss if global_step == 0 else (
                                                                                             tr_loss - logging_loss) / args.save_steps
                    logging_loss = tr_loss
                    logger.info(f"step:      {global_step},   train_loss: {train_loss}, "
                                f"eval_loss: {eval_loss},     eval_accu:  {eval_accu}, "
                                f"best_loss: {best_dev_loss}, best_accu:  {best_dev_acc}, best_step: {best_step}")

                global_step += 1

            if (args.max_steps > 0) and (global_step > args.max_steps):
                epoch_iterator.close()
                break

        # TODO trian and evaluate
        results = eval(args, model, dev_dataset, prefix="", test=False)
        train_loss = epoch_loss / epoch_step
        eval_loss = results["eval_loss"]
        eval_accu = results["eval_acc"]
        update_epoch_loss(train_loss, eval_loss, epoch)
        update_eval_accuracy(eval_accu, epoch)
        if eval_accu > best_dev_acc:
            best_dev_acc = eval_accu
            best_dev_loss = eval_loss
            best_step = global_step

            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(args.output_dir)
            model_path = os.path.join(args.output_dir, f'training_args.bin')
            torch.save(args, model_path)
            result_path = os.path.join(args.output_dir, f"eval_result_{global_step}.json")
            with open(result_path, "w", encoding="utf-8") as writer:
                json.dump(results, writer, ensure_ascii=False, indent=4)

        logger.info(f"epoch:     {epoch},         train_loss: {train_loss}, "
                    f"eval_loss: {eval_loss},     eval_accu:  {eval_accu}, "
                    f"best_loss: {best_dev_loss}, best_accu:  {best_dev_acc}, best_step: {best_step}")

        if (args.max_steps > 0) and (global_step > args.max_steps):
            train_iterator.close()
            break

    return global_step, tr_loss / global_step, best_step


def eval(args, model, eval_dataset, prefix="", test=False):
    eval_task_names = (args.task_name,)
    eval_output_dir = args.output_dir

    results = {}
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    # logger.info("***** Running evaluation {} *****".format(prefix))
    # logger.info("  Num examples = %d", len(eval_dataset))
    # logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        args.device = torch.device("cuda")
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    acc = simple_accuracy(preds, out_label_ids)
    result = {"eval_acc": acc, "eval_loss": eval_loss}
    results.update(result)

    # output_eval_file = os.path.join(eval_output_dir, "is_test_" + str(test).lower() + "_eval_results.txt")
    #
    # with open(output_eval_file, "w") as writer:
    #     logger.info("***** Eval results {} *****".format(str(prefix) + " is test:" + str(test)))
    #     writer.write("model           =%s\n" % str(args.model_name_or_path))
    #     writer.write("total batch size=%d\n" % (args.per_gpu_train_batch_size * args.gradient_accumulation_steps *
    #                                             (torch.distributed.get_world_size()
    #                                              if args.local_rank != -1 else 1)))
    #     writer.write("train num epochs=%d\n" % args.num_train_epochs)
    #     writer.write("fp16            =%s\n" % args.fp16)
    #     writer.write("max seq length  =%d\n" % args.max_seq_length)
    #     for key in sorted(result.keys()):
    #         logger.info("  %s = %s", key, str(result[key]))
    #         writer.write("%s = %s\n" % (key, str(result[key])))
    return results


def parse_args():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",            type=str, default="../../dataset_cosmosqa_json",
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type",          type=str, default="bert",
                        help="Model type selected in the list: ")
    parser.add_argument("--model_name_or_path",  type=str, default="../../pretrained_model/_roberta-base",
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--task_name",           type=str, default="cosmosqa",
                        help="The name of the task to train selected in the list: " + ", ".join(["cosmosqa"]))
    parser.add_argument("--output_dir",          type=str, default="output/",
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_choice",          type=str, default="base")
    parser.add_argument("--debug",               action="store_true")

    ## Other parameters
    parser.add_argument("--config_name",               default="",  type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name",            default="",  type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir",                 default="",  type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",            default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train",                  action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",                   action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",                   action='store_true',
                        help='Whether to run test on the test set')
    parser.add_argument("--evaluate_during_training",  action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case",             action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size",    default=8,    type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size",     default=8,    type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', default=1,    type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate",               default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",                default=0.0,  type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon",                default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",               default=1.0,  type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs",            default=5.0,  type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",                   default=-1,   type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps",                default=0,    type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps',        type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps',           type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending "
                             "and ending with step number")
    parser.add_argument("--no_cuda",              action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache',      action='store_true',
                        help="Overwrite the cached training and evaluation sets")

    parser.add_argument('--fp16',           action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--seed',           type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank",     type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip',      type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port',    type=str, default='', help="For distant debugging.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Set seed
    args.n_gpu = torch.cuda.device_count()

    if args.model_choice == "large":
        args.per_gpu_train_batch_size = 1
        args.per_gpu_eval_batch_size = 2
        args.learning_rate = 1e-5
        args.model_name_or_path = os.path.join(grg_dir, "pretrained_model/roberta-large")
    elif args.model_choice == "base":
        args.per_gpu_train_batch_size = 4
        args.per_gpu_eval_batch_size = 4
        args.learning_rate = 2e-5
        args.model_name_or_path = os.path.join(grg_dir, "pretrained_model/roberta-base")
    else:
        raise ValueError

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size  = args.per_gpu_eval_batch_size  * max(1, args.n_gpu)
    print("args.train_batch_size = ", args.train_batch_size)
    print("args.eval_batch_size = ", args.eval_batch_size)

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    args_path = os.path.join(args.output_dir, "args.json")
    with open(args_path, "w", encoding="utf-8") as writer:
        json.dump(args.__dict__, writer, ensure_ascii=False, indent=4)

    set_seed(args)
    processor  = CosmosProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    if args.model_choice == "base":
        from baseline_cosmosqa.model.model_tran_roberta.model_base import RobertaForMultipleChoice
    elif args.model_choice == "large":
        from baseline_cosmosqa.model.model_tran_roberta.model_large import RobertaForMultipleChoice
    else:
        raise ValueError

    config_class, model_class, tokenizer_class = RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, finetuning_task=args.task_name)

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)

    # model = model_class.from_pretrained(args.model_name_or_path,
    #                                     from_tf=bool('.ckpt' in args.model_name_or_path),
    #                                     config=config)
    model = model_class.from_pretrained(args.model_name_or_path)

    def convert(features):
        # Convert to Tensors and build dataset
        all_input_ids   = torch.tensor(select_field(features, 'input_ids'),   dtype=torch.long)
        all_input_mask  = torch.tensor(select_field(features, 'input_mask'),  dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_label_ids   = torch.tensor([f.label for f in features],           dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset

    if args.debug:
        print("debug")
        print("--- time: {} ---, get examples".format(time.ctime(time.time())))
        train_examples = processor.get_train_examples(args.data_dir)[:32]
        dev_examples   = processor.get_dev_examples(args.data_dir)[:32]

        print("--- time: {} ---, create features".format(time.ctime(time.time())))
        train_features = processor._create_features(train_examples,
                                                    tokenizer,
                                                    args.max_seq_length,
                                                    sep_token_extra=True)
        dev_features = processor._create_features(dev_examples,
                                                  tokenizer,
                                                  args.max_seq_length,
                                                  sep_token_extra=True)

        train_dataset = convert(train_features)
        dev_dataset = convert(dev_features)

    else:
        cached_train_features_file = "./train_features_{}.pkl".format(args.max_seq_length)
        cached_dev_features_file   = "./dev_features_{}.pkl".format(args.max_seq_length)

        try:
            print("--- time: {} ---, load features".format(time.ctime(time.time())))
            with open(cached_train_features_file, "rb") as reader:
                train_features = pickle.load(reader)
            with open(cached_dev_features_file, "rb") as reader:
                dev_features = pickle.load(reader)

        except:
            print("--- time: {} ---, get examples".format(time.ctime(time.time())))
            train_examples = processor.get_train_examples(args.data_dir)
            dev_examples   = processor.get_dev_examples(args.data_dir)

            print("--- time: {} ---, create features".format(time.ctime(time.time())))
            train_features = processor._create_features(train_examples,
                                                        tokenizer,
                                                        args.max_seq_length,
                                                        sep_token_extra=True)
            dev_features   = processor._create_features(dev_examples,
                                                        tokenizer,
                                                        args.max_seq_length,
                                                        sep_token_extra=True)

            print("--- time: {} ---, dump features".format(time.ctime(time.time())))
            with open(cached_train_features_file, "wb") as writer:
                pickle.dump(train_features, writer)
            with open(cached_dev_features_file, "wb") as writer:
                pickle.dump(dev_features, writer)

        train_dataset  = convert(train_features)
        dev_dataset    = convert(dev_features)

    # Training
    if args.do_train:
        global_step, tr_loss, best_step = train(args, train_dataset, dev_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s, best_step = %s", global_step, tr_loss, best_step)


if __name__ == "__main__":
    main()

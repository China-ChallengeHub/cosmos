from __future__ import absolute_import, division, print_function

import os
import sys
import time
import json
import torch
import random
import logging
import numpy as np

cur_path = os.path.abspath(__file__)
cur_dir = os.path.dirname(cur_path)
par_dir = os.path.dirname(cur_dir)
gra_dir = os.path.dirname(par_dir)
grg_dir = os.path.dirname(gra_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)
sys.path.append(gra_dir)
sys.path.append(grg_dir)

from tqdm import tqdm
from tqdm import trange
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_xbert import RobertaConfig
from pytorch_pretrained_xbert import RobertaTokenizer
from pytorch_pretrained_xbert import RobertaForMultipleChoice

from pytorch_pretrained_xbert import AdamW
from pytorch_pretrained_xbert import WarmupLinearSchedule

from baseline_cosmosqa.run_roberta.config import parse_args
from baseline_cosmosqa.run_roberta.utils_cosmosqa import read_feature
from baseline_cosmosqa.run_roberta.visual_util import update_lr
from baseline_cosmosqa.run_roberta.visual_util import update_epoch_loss
from baseline_cosmosqa.run_roberta.visual_util import update_step_result
from baseline_cosmosqa.run_roberta.visual_util import update_eval_accuracy

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def init_optimizer(args, train_dataloader, model):
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.warmup_steps = int(t_total * 0.1)
    # args.warmup_steps = 1000
    print("args.warmup_steps = ", args.warmup_steps)

    args_path = os.path.join(args.output_dir, "args.json")
    with open(args_path, "w", encoding="utf-8") as writer:
        json.dump(args.__dict__, writer, ensure_ascii=False, indent=4)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    model.cuda()
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        gpu_ids = list(range(args.n_gpu))
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    # # Distributed training (should be after apex fp16 initialization)
    # if args.local_rank != -1:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
    #                                                       output_device=args.local_rank,
    #                                                       find_unused_parameters=True)

    return model, optimizer, scheduler, t_total


def train(args, train_dataset, dev_dataset, model, tokenizer):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    model, optimizer, scheduler, t_total = init_optimizer(args, train_dataloader, model)

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
                      'token_type_ids': None,
                      'labels':         batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

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
                        best_dev_acc  = eval_accu
                        best_dev_loss = eval_loss
                        best_step     = global_step

                        model_to_save = model.module if hasattr(model, 'module') else model
                        model_to_save.save_pretrained(args.output_dir)
                        model_path = os.path.join(args.output_dir, f'training_args.bin')
                        torch.save(args, model_path)
                        result_path = os.path.join(args.output_dir, f"eval_result_{global_step}.json")
                        with open(result_path, "w", encoding="utf-8") as writer:
                            json.dump(results, writer, ensure_ascii=False, indent=4)

                    train_loss = tr_loss - logging_loss if global_step == 0 else (tr_loss - logging_loss) / args.save_steps
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
            best_dev_acc  = eval_accu
            best_dev_loss = eval_loss
            best_step     = global_step

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
                      'token_type_ids': None,
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

    return results


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
        args.per_gpu_eval_batch_size = 8
        args.model_name_or_path = os.path.join(grg_dir, "pretrained_model/roberta-base")
    else:
        raise ValueError

    num_labels = 4
    print("[TIME] --- time: {} ---, prepare model".format(time.ctime(time.time())))
    config_class, model_class, tokenizer_class = RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, finetuning_task=args.task_name)

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)
    print("[TIME] --- time: {} ---, prepared model".format(time.ctime(time.time())))

    train_dataset, dev_dataset = read_feature(args, tokenizer)
    # Training
    if args.do_train:
        global_step, tr_loss, best_step = train(args, train_dataset, dev_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s, best_step = %s", global_step, tr_loss, best_step)


if __name__ == "__main__":
    main()

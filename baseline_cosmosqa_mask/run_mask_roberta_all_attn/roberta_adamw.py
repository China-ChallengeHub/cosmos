from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import torch
import random
import logging
import numpy as np

from tqdm import tqdm
from tqdm import trange
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
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

from pytorch_pretrained_xbert import AdamW
from pytorch_pretrained_xbert import WarmupLinearSchedule

from baseline_cosmosqa_mask.run_mask_roberta_all_attn.util_visual import update_lr
from baseline_cosmosqa_mask.run_mask_roberta_all_attn.util_visual import update_epoch_loss
from baseline_cosmosqa_mask.run_mask_roberta_all_attn.util_visual import update_step_result
from baseline_cosmosqa_mask.run_mask_roberta_all_attn.util_visual import update_eval_accuracy

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)       # 为当前CPU设置随机种子
    torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子

    # set default tensor type
    # torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)  # 为所有GPU设置随机种子
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark     = False


def init_optimizer(args, train_dataloader, model):
    args.num_train_epochs = 5

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # debug 2:
    args.t_total = t_total
    args.warmup_steps = 0
    # args.warmup_steps = 1000
    # args.warmup_steps = 1500
    # args.warmup_steps = int(t_total * 0.1)

    # debug 3:
    args.adam_epsilon = 1e-6
    # args.adam_epsilon = 1e-7
    # args.adam_epsilon = 1e-8
    # args.adam_epsilon = 1e-9

    # debug 4:
    # args.weight_decay = 0.1
    args.weight_decay = 0.05
    # args.weight_decay = 0.02
    # args.weight_decay = 0.01
    # args.weight_decay = 0

    # debug 5:
    # args.learning_rate = 5e-6
    args.learning_rate = 1e-5
    # args.learning_rate = 2e-5
    # args.learning_rate = 3e-5

    # debug 6:
    # args.seed = 42
    # args.seed = 5233

    print("args.num_train_epochs = ", args.num_train_epochs)
    print("args.t_total = ",          args.t_total)
    print("args.train_batch_size = ", args.train_batch_size)
    print("args.eval_batch_size = ",  args.eval_batch_size)
    print("args.n_gpu = ",            args.n_gpu)
    print("args.seed = ",             args.seed)
    print("args.warmup_steps = ",     args.warmup_steps)
    print("args.adam_epsilon = ",     args.adam_epsilon)
    print("args.weight_decay = ",     args.weight_decay)
    print("args.learning_rate = ",    args.learning_rate)
    print("args.max_grad_norm = ",    args.max_grad_norm)

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

    return model, optimizer, scheduler, t_total


def train(args, train_dataset, dev_dataset, model):
    """ Train the model """
    train_sampler    = RandomSampler(train_dataset)
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
            inputs = {'input_ids':        batch[0],
                      'attention_mask':   batch[1],
                      'token_type_ids':   batch[2],
                      'commonsense_mask': batch[3].long(),
                      'dependency_mask':  batch[4].long(),
                      'entity_mask':      batch[5].long(),
                      'sentiment_mask':   batch[6].long(),
                      'labels':           batch[7]}
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
                    results = eval(args, model, dev_dataset)
                    results["learning_rate"] = lr_this_step

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
        lr_this_step = scheduler.get_lr()[0]
        update_lr(lr_this_step, global_step)

        results = eval(args, model, dev_dataset)
        results["learning_rate"] = lr_this_step

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


def eval(args, model, eval_dataset):
    results = {}
    eval_sampler    = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        args.device = torch.device("cuda")
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':        batch[0],
                      'attention_mask':   batch[1],
                      'token_type_ids':   batch[2],
                      'commonsense_mask': batch[3].long(),
                      'dependency_mask':  batch[4].long(),
                      'entity_mask':      batch[5].long(),
                      'sentiment_mask':   batch[6].long(),
                      'labels':           batch[7]}

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

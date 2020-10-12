from __future__ import absolute_import, division, print_function

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

from baseline_cosmosqa_mask.run_mask_roberta_all_new.util_visual import update_lr
from baseline_cosmosqa_mask.run_mask_roberta_all_new.util_visual import update_epoch_loss
from baseline_cosmosqa_mask.run_mask_roberta_all_new.util_visual import update_step_result
from baseline_cosmosqa_mask.run_mask_roberta_all_new.util_visual import update_eval_accuracy

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
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, dev_dataset, model, tokenizer):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    print("learning_rate = ", args.learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.cuda()
    if args.n_gpu > 1:
        gpu_ids = list(range(args.n_gpu))
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    # Train!
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

            # # TODO debug3: 训练过程中不更改学习率
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            tr_loss += loss.item()
            tr_step += 1

            epoch_loss += loss.item()
            epoch_step += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                lr_this_step = args.learning_rate
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
            inputs = {'input_ids':        batch[0],
                      'attention_mask':   batch[1],
                      'token_type_ids':   batch[2],
                      'commonsense_mask': batch[3].long(),
                      'dependency_mask':  batch[4].long(),
                      'entity_mask':      batch[5].long(),
                      'sentiment_mask':         batch[6].long(),
                      'labels':           batch[7]}

            # print("inputs keys = ", inputs.keys())
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

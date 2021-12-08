# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from collections import deque
import shutil

import numpy as np
import paddle
import paddle.nn.functional as F

from paddleseg.utils import (TimeAverager, calculate_eta, resume, logger,
                             worker_init_fn, train_profiler, op_flops_funs)
from paddleseg.core.val import evaluate
from paddleslim import UnstructuredPruner, GMPUnstructuredPruner


def check_logits_losses(logits_list, losses):
    len_logits = len(logits_list)
    len_losses = len(losses['types'])
    if len_logits != len_losses:
        raise RuntimeError(
            'The length of logits_list should equal to the types of loss config: {} != {}.'
            .format(len_logits, len_losses))


def loss_computation(logits_list, labels, losses, edges=None):
    check_logits_losses(logits_list, losses)
    loss_list = []
    for i in range(len(logits_list)):
        logits = logits_list[i]
        loss_i = losses['types'][i]
        # Whether to use edges as labels According to loss type.
        if loss_i.__class__.__name__ in ('BCELoss',
                                         'FocalLoss') and loss_i.edge_label:
            loss_list.append(losses['coef'][i] * loss_i(logits, edges))
        elif loss_i.__class__.__name__ in ("KLLoss", ):
            loss_list.append(losses['coef'][i] * loss_i(
                logits_list[0], logits_list[1].detach()))
        else:
            loss_list.append(losses['coef'][i] * loss_i(logits, labels))
    return loss_list


def create_unstructured_pruner(model, args, configs=None):
    if configs is None:
        return UnstructuredPruner(
            model,
            mode=args.pruning_mode,
            ratio=args.ratio,
            threshold=args.threshold,
            prune_params_type=args.prune_params_type,
            local_sparsity=args.local_sparsity)
    else:
        return GMPUnstructuredPruner(
            model,
            ratio=args.ratio,
            prune_params_type=args.prune_params_type,
            local_sparsity=args.local_sparsity,
            configs=configs)


def train(args,
          model,
          train_dataset,
          val_datasets=None,
          class_weights=None,
          dataset_weights=None,
          optimizer=None,
          save_dir='output',
          iters=10000,
          batch_size=2,
          resume_model=None,
          save_interval=1000,
          log_iters=10,
          num_workers=0,
          use_vdl=False,
          losses=None,
          keep_checkpoint_max=5):
    """
    Launch training.

    Args:
        model（nn.Layer): A sementic segmentation model.
        train_dataset (paddle.io.Dataset): Used to read and process training datasets.
        val_datasets (paddle.io.Dataset, optional): Used to read and process validation datasets.
        optimizer (paddle.optimizer.Optimizer): The optimizer.
        save_dir (str, optional): The directory for saving the model snapshot. Default: 'output'.
        iters (int, optional): How may iters to train the model. Defualt: 10000.
        batch_size (int, optional): Mini batch size of one gpu or cpu. Default: 2.
        resume_model (str, optional): The path of resume model.
        save_interval (int, optional): How many iters to save a model snapshot once during training. Default: 1000.
        log_iters (int, optional): Display logging information at every log_iters. Default: 10.
        num_workers (int, optional): Num workers for data loader. Default: 0.
        use_vdl (bool, optional): Whether to record the data to VisualDL during training. Default: False.
        losses (dict): A dict including 'types' and 'coef'. The length of coef should equal to 1 or len(losses['types']).
            The 'types' item is a list of object of paddleseg.models.losses while the 'coef' item is a list of the relevant coefficient.
        keep_checkpoint_max (int, optional): Maximum number of checkpoints to save. Default: 5.
    """
    model.train()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank

    start_iter = 0
    if val_datasets is not None:
        assert class_weights is not None
        assert dataset_weights is not None
    if resume_model is not None:
        start_iter = resume(model, optimizer, resume_model)

    if not os.path.isdir(save_dir):
        if os.path.exists(save_dir):
            os.remove(save_dir)
        os.makedirs(save_dir)

    if nranks > 1:
        paddle.distributed.fleet.init(is_collective=True)
        optimizer = paddle.distributed.fleet.distributed_optimizer(
            optimizer)  # The return is Fleet object
        ddp_model = paddle.distributed.fleet.distributed_model(model)

    batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True,
        worker_init_fn=worker_init_fn,
    )

    if use_vdl:
        from visualdl import LogWriter
        log_writer = LogWriter(save_dir)

    avg_loss = 0.0
    avg_loss_list = []
    iters_per_epoch = len(batch_sampler)
    best_total_iou = -1.0
    best_class_ious = []
    best_dataset_ious = []
    best_model_iter = -1
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    save_models = deque()
    batch_start = time.time()
    iter = start_iter

    if args.pruning_strategy == 'gmp':
        # GMP pruner step 0: define configs. No need to do this if you are not using 'gmp'
        configs = {
            'stable_iterations': args.stable_epochs * iters_per_epoch,  # 0
            'pruning_iterations': args.pruning_epochs *
            iters_per_epoch,  # original_total_iters * 0.4~0.45
            'tunning_iterations': args.tunning_epochs *
            iters_per_epoch,  # original_total_iters * 0.4~0.45
            'resume_iteration':
            (args.last_epoch + 1) * iters_per_epoch,  # args.last_epoch=-1
            'pruning_steps': args.pruning_steps,  # args.pruning_epochs * 2
            'initial_ratio': args.initial_ratio,  # 0.15
        }
    else:
        configs = None

    # GMP pruner step 1: initialize a pruner object
    pruner = create_unstructured_pruner(model, args, configs=configs)

    while iter < iters:
        for data in loader:
            iter += 1
            if iter > iters:
                break
            reader_cost_averager.record(time.time() - batch_start)
            images = data[0]
            labels = data[1].astype('int64')
            edges = None
            if len(data) == 3:
                edges = data[2].astype('int64')

            if iter % iters_per_epoch == 0 and hasattr(train_dataset,
                                                       'shuffle'):
                train_dataset.shuffle()
                logger.info('Shuffle train dataset')

            if nranks > 1:
                logits_list = ddp_model(images)
            else:
                logits_list = model(images)
            loss_list = loss_computation(
                logits_list=logits_list,
                labels=labels,
                losses=losses,
                edges=edges)
            loss = sum(loss_list)
            loss.backward()

            optimizer.step()
            lr = optimizer.get_lr()
            # update lr
            if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                lr_sche = optimizer.user_defined_optimizer._learning_rate
            else:
                lr_sche = optimizer._learning_rate
            if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
                lr_sche.step()

            model.clear_gradients()

            # GMP pruner step 2: step() to update ratios and other internal states of the pruner.
            pruner.step()

            avg_loss += loss.numpy()[0]
            if not avg_loss_list:
                avg_loss_list = [l.numpy() for l in loss_list]
            else:
                for i in range(len(loss_list)):
                    avg_loss_list[i] += loss_list[i].numpy()
            batch_cost_averager.record(
                time.time() - batch_start, num_samples=batch_size)

            if (iter) % log_iters == 0 and local_rank == 0:
                avg_loss /= log_iters
                avg_loss_list = [l[0] / log_iters for l in avg_loss_list]
                remain_iters = iters - iter
                avg_train_batch_cost = batch_cost_averager.get_average()
                avg_train_reader_cost = reader_cost_averager.get_average()
                eta = calculate_eta(remain_iters, avg_train_batch_cost)
                logger.info(
                    "[TRAIN] epoch: {}, iter: {}/{}, loss: {:.4f}, lr: {:.6f}, batch_cost: {:.4f}, reader_cost: {:.5f}, ips: {:.4f} samples/sec | ETA {}"
                    .format((iter - 1) // iters_per_epoch + 1, iter, iters,
                            avg_loss, lr, avg_train_batch_cost,
                            avg_train_reader_cost,
                            batch_cost_averager.get_ips_average(), eta))
                if use_vdl:
                    log_writer.add_scalar('Train/loss', avg_loss, iter)
                    # Record all losses if there are more than 2 losses.
                    if len(avg_loss_list) > 1:
                        avg_loss_dict = {}
                        for i, value in enumerate(avg_loss_list):
                            avg_loss_dict['loss_' + str(i)] = value
                        for key, value in avg_loss_dict.items():
                            log_tag = 'Train/' + key
                            log_writer.add_scalar(log_tag, value, iter)

                    log_writer.add_scalar('Train/lr', lr, iter)
                    log_writer.add_scalar('Train/batch_cost',
                                          avg_train_batch_cost, iter)
                    log_writer.add_scalar('Train/reader_cost',
                                          avg_train_reader_cost, iter)
                avg_loss = 0.0
                avg_loss_list = []
                reader_cost_averager.reset()
                batch_cost_averager.reset()

            if (iter % save_interval == 0
                    or iter == iters) and (val_datasets is not None):
                num_workers = 1 if num_workers > 0 else 0
                class_ious = []
                for val_dataset in val_datasets:
                    # GMP pruner step 3: update params before summrizing sparsity, saving model or evaluation.
                    pruner.update_params()
                    logger.info(
                        "The current sparsity of the pruned model is: {}%".
                        format(
                            round(100 * UnstructuredPruner.total_sparse(model),
                                  2)))
                    mean_iou, acc, class_iou, _, _ = evaluate(
                        model, val_dataset, num_workers=num_workers)
                    class_ious.append(class_iou)
                model.train()

            if (iter % save_interval == 0 or iter == iters) and local_rank == 0:
                current_save_dir = os.path.join(save_dir,
                                                "iter_{}".format(iter))
                if not os.path.isdir(current_save_dir):
                    os.makedirs(current_save_dir)
                # GMP pruner step 3: update params before summrizing sparsity, saving model or evaluation.
                pruner.update_params()

                paddle.save(model.state_dict(),
                            os.path.join(current_save_dir, 'model.pdparams'))
                paddle.save(optimizer.state_dict(),
                            os.path.join(current_save_dir, 'model.pdopt'))
                save_models.append(current_save_dir)
                if len(save_models) > keep_checkpoint_max > 0:
                    model_to_remove = save_models.popleft()
                    shutil.rmtree(model_to_remove)

                if val_datasets is not None:
                    dataset_ious = []
                    for class_iou in class_ious:
                        dataset_iou = 0
                        for num, class_weight in enumerate(class_weights):
                            dataset_iou += class_weight * class_iou[num]
                        dataset_ious.append(dataset_iou)
                    total_iou = 0
                    for num, dataset_iou in enumerate(dataset_ious):
                        logger.info("[EVAL] Dataset {} Class IoU: {}\n".format(
                            num, str(np.round(class_ious[num], 4))))
                        logger.info("[EVAL] Dataset {} IoU: {}\n".format(
                            num, str(np.round(dataset_iou, 4))))
                        total_iou += dataset_weights[num] * dataset_iou
                    logger.info("[EVAL] Total IoU: \n" +
                                str(np.round(total_iou, 4)))

                    if args.pruning_strategy == 'gmp':
                        save_best = iter >= configs[
                            'stable_iterations'] + configs['pruning_iterations']
                    else:
                        save_best = True
                    if save_best:
                        if total_iou > best_total_iou:
                            best_total_iou = total_iou
                            best_class_ious = class_ious
                            best_dataset_ious = dataset_ious
                            best_model_iter = iter
                            best_model_dir = os.path.join(
                                save_dir, "best_model")
                            paddle.save(
                                model.state_dict(),
                                os.path.join(best_model_dir, 'model.pdparams'))
                        logger.info(
                            '====================== best model metrics =================='
                        )
                        for num, best_class_iou in enumerate(best_class_ious):
                            logger.info(
                                "[EVAL] Dataset {} Class IoU: {}".format(
                                    num, str(np.round(best_class_iou, 4))))
                            logger.info("[EVAL] Dataset {} IoU: {}\n".format(
                                num, str(np.round(best_dataset_ious[num], 4))))
                        logger.info("[EVAL] Total IoU: \n" +
                                    str(np.round(total_iou, 4)))
                        logger.info(
                            '[EVAL] The best model was saved at iter {}.'.
                            format(best_model_iter))
                        logger.info(
                            '==========================================================='
                        )

                    if use_vdl:
                        log_writer.add_scalar('Evaluate/total IoU', total_iou,
                                              iter)
                        for k, class_iou in enumerate(class_ious):
                            for i, iou in enumerate(class_iou):
                                log_writer.add_scalar(
                                    'Evaluate/Dataset {} IoU {}'.format(k, i),
                                    float(iou), iter)

            batch_start = time.time()

    # Calculate flops.
    if local_rank == 0:
        _, c, h, w = images.shape
        _ = paddle.flops(
            model, [1, c, h, w],
            custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})

    # Sleep for half a second to let dataloader release resources.
    time.sleep(0.5)
    if use_vdl:
        log_writer.close()

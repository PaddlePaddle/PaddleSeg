# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import shutil
import numpy as np
from collections import deque

import paddle
import paddle.nn.functional as F

from medicalseg.utils import (TimeAverager, calculate_eta, resume, logger,
                              worker_init_fn, train_profiler, op_flops_run,
                              loss_computation)
from medicalseg.core.val import evaluate


def train(model,
          train_dataset,
          val_dataset=None,
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
          keep_checkpoint_max=5,
          profiler_options=None,
          to_static_training=False,
          sw_num=None,
          is_save_data=True,
          has_dataset_json=True):
    """
    Launch training.

    Args:
        model（nn.Layer): A sementic segmentation model.
        train_dataset (paddle.io.Dataset): Used to read and process training datasets.
        val_dataset (paddle.io.Dataset, optional): Used to read and process validation datasets.
        optimizer (paddle.optimizer.Optimizer): The optimizer.
        save_dir (str, optional): The directory for saving the model snapshot. Default: 'output'.
        iters (int, optional): How may iters to train the model. Defualt: 10000.
        batch_size (int, optional): Mini batch size of one gpu or cpu. Default: 2.
        resume_model (str, optional): The path of resume model.
        save_interval (int, optional): How many iters to save a model snapshot once during training. Default: 1000.
        log_iters (int, optional): Display logging information at every log_iters. Default: 10.
        num_workers (int, optional): Num workers for data loader. Default: 0.
        use_vdl (bool, optional): Whether to record the data to VisualDL during training. Default: False.
        losses (dict, optional): A dict including 'types' and 'coef'. The length of coef should equal to 1 or len(losses['types']).
            The 'types' item is a list of object of medseg.models.losses while the 'coef' item is a list of the relevant coefficient.
        keep_checkpoint_max (int, optional): Maximum number of checkpoints to save. Default: 5.
        profiler_options (str, optional): The option of train profiler.
        to_static_training (bool, optional): Whether to use @to_static for training.
        sw_num:sw batch size.
        is_save_data:use savedata function
        has_dataset_json:has dataset_json
    """
    model.train()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank

    start_iter = 0
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
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True,
        worker_init_fn=worker_init_fn, )

    if use_vdl:
        from visualdl import LogWriter
        log_writer = LogWriter(save_dir)
    else:
        log_writer = None

    if to_static_training:
        model = paddle.jit.to_static(model)
        logger.info("Successfully to apply @to_static")

    avg_loss = 0.0
    avg_loss_list = []
    mdice = 0.0
    channel_dice_array = np.array([])
    iters_per_epoch = len(batch_sampler)
    best_mean_dice = -1.0
    best_model_iter = -1
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    save_models = deque()
    batch_start = time.time()

    iter = start_iter
    while iter < iters:
        for data in loader:
            reader_cost_averager.record(time.time() - batch_start)
            images = data[0]
            labels = data[1].astype('int32')

            if hasattr(model, 'data_format') and model.data_format == 'NDHWC':
                images = images.transpose((0, 2, 3, 4, 1))

            if nranks > 1:
                logits_list = ddp_model(images)
            else:
                logits_list = model(images)

            # label.shape: (num_class, D, H, W) logit.shape: (N, num_class, D, H, W)
            loss_list, per_channel_dice = loss_computation(
                logits_list=logits_list, labels=labels, losses=losses)
            loss = sum(loss_list)

            loss.backward()  # grad is nan when set elu=True
            optimizer.step()

            lr = optimizer.get_lr()
            iter += 1

            # update lr
            if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                lr_sche = optimizer.user_defined_optimizer._learning_rate
            else:
                lr_sche = optimizer._learning_rate
            if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
                lr_sche.step()

            train_profiler.add_profiler_step(profiler_options)

            model.clear_gradients()
            # TODO use a function to record, print lossetc

            avg_loss += loss.numpy()[0]
            mdice += np.mean(per_channel_dice) * 100

            if channel_dice_array.size == 0:
                channel_dice_array = per_channel_dice
            else:
                channel_dice_array += per_channel_dice

            if len(avg_loss_list) == 0:
                avg_loss_list = [l.numpy() for l in loss_list]
            else:
                for i in range(len(loss_list)):
                    avg_loss_list[i] += loss_list[i].numpy()

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=batch_size)

            if (iter) % log_iters == 0 and local_rank == 0:
                avg_loss /= log_iters
                avg_loss_list = [l[0] / log_iters for l in avg_loss_list]
                mdice /= log_iters
                channel_dice_array = channel_dice_array / log_iters

                remain_iters = iters - iter
                avg_train_batch_cost = batch_cost_averager.get_average()
                avg_train_reader_cost = reader_cost_averager.get_average()
                eta = calculate_eta(remain_iters, avg_train_batch_cost)
                logger.info(
                    "[TRAIN] epoch: {}, iter: {}/{}, loss: {:.4f}, DSC: {:.4f}, "
                    "lr: {:.6f}, batch_cost: {:.4f}, reader_cost: {:.5f}, ips: {:.4f} samples/sec | ETA {}"
                    .format((iter
                             ) // iters_per_epoch, iter, iters, avg_loss, mdice,
                            lr, avg_train_batch_cost, avg_train_reader_cost,
                            batch_cost_averager.get_ips_average(), eta))

                if use_vdl:
                    log_writer.add_scalar('Train/loss', avg_loss, iter)
                    # Record all losses if there are more than 2 losses.
                    if len(avg_loss_list) > 1:
                        for i, loss in enumerate(avg_loss_list):
                            log_writer.add_scalar('Train/loss_{}'.format(i),
                                                  loss, iter)

                    log_writer.add_scalar('Train/mdice', mdice, iter)
                    log_writer.add_scalar('Train/lr', lr, iter)
                    log_writer.add_scalar('Train/batch_cost',
                                          avg_train_batch_cost, iter)
                    log_writer.add_scalar('Train/reader_cost',
                                          avg_train_reader_cost, iter)
                avg_loss = 0.0
                avg_loss_list = []
                mdice = 0.0
                channel_dice_array = np.array([])
                reader_cost_averager.reset()
                batch_cost_averager.reset()

            if (iter % save_interval == 0 or iter == iters) and (
                    val_dataset is not None):
                num_workers = 1 if num_workers > 0 else 0

                result_dict = evaluate(
                    model,
                    val_dataset,
                    losses,
                    num_workers=num_workers,
                    writer=log_writer,
                    print_detail=True,
                    auc_roc=False,
                    save_dir=save_dir,
                    sw_num=sw_num,
                    is_save_data=is_save_data,
                    has_dataset_json=has_dataset_json)

                model.train()

            if (iter % save_interval == 0 or iter == iters) and local_rank == 0:
                current_save_dir = os.path.join(save_dir,
                                                "iter_{}".format(iter))
                if not os.path.isdir(current_save_dir):
                    os.makedirs(current_save_dir)
                paddle.save(model.state_dict(),
                            os.path.join(current_save_dir, 'model.pdparams'))
                paddle.save(optimizer.state_dict(),
                            os.path.join(current_save_dir, 'model.pdopt'))
                save_models.append(current_save_dir)
                if len(save_models) > keep_checkpoint_max > 0:
                    model_to_remove = save_models.popleft()
                    shutil.rmtree(model_to_remove)

                if val_dataset is not None:
                    if result_dict['mdice'] > best_mean_dice:
                        best_mean_dice = result_dict['mdice']
                        best_model_iter = iter
                        best_model_dir = os.path.join(save_dir, "best_model")
                        paddle.save(
                            model.state_dict(),
                            os.path.join(best_model_dir, 'model.pdparams'))
                    logger.info(
                        '[EVAL] The model with the best validation mDice ({:.4f}) was saved at iter {}.'
                        .format(best_mean_dice, best_model_iter))

                    if use_vdl:
                        log_writer.add_scalar('Evaluate/Dice',
                                              result_dict['mdice'], iter)
                        if "auc_roc" in result_dict:
                            log_writer.add_scalar('Evaluate/auc_roc',
                                                  result_dict['auc_roc'], iter)

            batch_start = time.time()

    if local_rank == 0:
        _, c, d, h, w = images.shape
        _ = paddle.flops(
            model, [1, c, d, h, w],
            custom_ops={paddle.nn.SyncBatchNorm: op_flops_run.count_syncbn})

    # Sleep for half a second to let dataloader release resources.
    time.sleep(0.5)
    if use_vdl:
        log_writer.close()

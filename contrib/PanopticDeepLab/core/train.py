# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn.functional as F
from paddleseg.utils import TimeAverager, calculate_eta, resume, logger

from core.val import evaluate


def check_logits_losses(logits_list, losses):
    len_logits = len(logits_list)
    len_losses = len(losses['types'])
    if len_logits != len_losses:
        raise RuntimeError(
            'The length of logits_list should equal to the types of loss config: {} != {}.'
            .format(len_logits, len_losses))


def loss_computation(logits_list, semantic, semantic_weights, center,
                     center_weights, offset, offset_weights, losses):
    # semantic loss
    semantic_loss = losses['types'][0](logits_list[0], semantic,
                                       semantic_weights)
    semantic_loss = semantic_loss * losses['coef'][0]

    # center loss
    center_loss = losses['types'][1](logits_list[1], center)
    center_weights = (center_weights.unsqueeze(1)).expand_as(center_loss)
    center_loss = center_loss * center_weights
    if center_loss.sum() > 0:
        center_loss = center_loss.sum() / center_weights.sum()
    else:
        center_loss = center_loss.sum() * 0
    center_loss = center_loss * losses['coef'][1]

    # offset loss
    offset_loss = losses['types'][2](logits_list[2], offset)
    offset_weights = (offset_weights.unsqueeze(1)).expand_as(offset_loss)
    offset_loss = offset_loss * offset_weights
    if offset_weights.sum() > 0:
        offset_loss = offset_loss.sum() / offset_weights.sum()
    else:
        offset_loss = offset_loss.sum() * 0
    offset_loss = offset_loss * losses['coef'][2]

    loss_list = [semantic_loss, center_loss, offset_loss]

    return loss_list


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
          threshold=0.1,
          nms_kernel=7,
          top_k=200):
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
        losses (dict): A dict including 'types' and 'coef'. The length of coef should equal to 1 or len(losses['types']).
            The 'types' item is a list of object of paddleseg.models.losses while the 'coef' item is a list of the relevant coefficient.
        keep_checkpoint_max (int, optional): Maximum number of checkpoints to save. Default: 5.
        threshold (float, optional): A Float, threshold applied to center heatmap score. Default: 0.1.
        nms_kernel (int, optional): An Integer, NMS max pooling kernel size. Default: 7.
        top_k (int, optional): An Integer, top k centers to keep. Default: 200.
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
        # Initialize parallel environment if not done.
        if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
        ):
            paddle.distributed.init_parallel_env()
            ddp_model = paddle.DataParallel(model)
        else:
            ddp_model = paddle.DataParallel(model)

    batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True,
    )

    if use_vdl:
        from visualdl import LogWriter
        log_writer = LogWriter(save_dir)

    avg_loss = 0.0
    avg_loss_list = []
    iters_per_epoch = len(batch_sampler)
    best_pq = -1.0
    best_model_iter = -1
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    save_models = deque()
    batch_start = time.time()

    iter = start_iter
    while iter < iters:
        for data in loader:
            iter += 1
            if iter > iters:
                break
            reader_cost_averager.record(time.time() - batch_start)
            images = data[0]
            semantic = data[1]
            semantic_weights = data[2]
            center = data[3]
            center_weights = data[4]
            offset = data[5]
            offset_weights = data[6]
            foreground = data[7]

            if nranks > 1:
                logits_list = ddp_model(images)
            else:
                logits_list = model(images)

            loss_list = loss_computation(
                logits_list=logits_list,
                losses=losses,
                semantic=semantic,
                semantic_weights=semantic_weights,
                center=center,
                center_weights=center_weights,
                offset=offset,
                offset_weights=offset_weights)
            loss = sum(loss_list)
            loss.backward()

            optimizer.step()
            lr = optimizer.get_lr()
            if isinstance(optimizer._learning_rate,
                          paddle.optimizer.lr.LRScheduler):
                optimizer._learning_rate.step()
            model.clear_gradients()
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
                    "[TRAIN] epoch={}, iter={}/{}, loss={:.4f}, lr={:.6f}, batch_cost={:.4f}, reader_cost={:.5f}, ips={:.4f} samples/sec | ETA {}"
                    .format((iter - 1) // iters_per_epoch + 1, iter, iters,
                            avg_loss, lr, avg_train_batch_cost,
                            avg_train_reader_cost,
                            batch_cost_averager.get_ips_average(), eta))
                logger.info(
                    "[LOSS] loss={:.4f}, semantic_loss={:.4f}, center_loss={:.4f}, offset_loss={:.4f}"
                    .format(avg_loss, avg_loss_list[0], avg_loss_list[1],
                            avg_loss_list[2]))
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

            # save model
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

            # eval model
            if (iter % save_interval == 0 or iter == iters) and (
                    val_dataset is
                    not None) and local_rank == 0 and iter > iters // 2:
                num_workers = 1 if num_workers > 0 else 0
                panoptic_results, semantic_results, instance_results = evaluate(
                    model,
                    val_dataset,
                    threshold=threshold,
                    nms_kernel=nms_kernel,
                    top_k=top_k,
                    num_workers=num_workers,
                    print_detail=False)
                pq = panoptic_results['pan_seg']['All']['pq']
                miou = semantic_results['sem_seg']['mIoU']
                map = instance_results['ins_seg']['mAP']
                map50 = instance_results['ins_seg']['mAP50']
                logger.info(
                    "[EVAL] PQ: {:.4f}, mIoU: {:.4f}, mAP: {:.4f}, mAP50: {:.4f}"
                    .format(pq, miou, map, map50))
                model.train()

            # save best model and add evaluate results to vdl
            if (iter % save_interval == 0 or iter == iters) and local_rank == 0:
                if val_dataset is not None and iter > iters // 2:
                    if pq > best_pq:
                        best_pq = pq
                        best_model_iter = iter
                        best_model_dir = os.path.join(save_dir, "best_model")
                        paddle.save(
                            model.state_dict(),
                            os.path.join(best_model_dir, 'model.pdparams'))
                    logger.info(
                        '[EVAL] The model with the best validation pq ({:.4f}) was saved at iter {}.'
                        .format(best_pq, best_model_iter))

                    if use_vdl:
                        log_writer.add_scalar('Evaluate/PQ', pq, iter)
                        log_writer.add_scalar('Evaluate/mIoU', miou, iter)
                        log_writer.add_scalar('Evaluate/mAP', map, iter)
                        log_writer.add_scalar('Evaluate/mAP50', map50, iter)
            batch_start = time.time()

    # Calculate flops.
    if local_rank == 0:

        def count_syncbn(m, x, y):
            x = x[0]
            nelements = x.numel()
            m.total_ops += int(2 * nelements)

        _, c, h, w = images.shape
        flops = paddle.flops(
            model, [1, c, h, w],
            custom_ops={paddle.nn.SyncBatchNorm: count_syncbn})

    # Sleep for half a second to let dataloader release resources.
    time.sleep(0.5)
    if use_vdl:
        log_writer.close()

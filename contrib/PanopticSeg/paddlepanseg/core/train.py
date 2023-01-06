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
import shutil
from collections import deque

import paddle
from paddleseg.utils import (TimeAverager, calculate_eta, resume, logger,
                             worker_init_fn, op_flops_funs)

from paddlepanseg.core.val import evaluate


def loss_computation(sample_dict, net_out_dict, losses):
    loss_list = []
    for i in range(len(losses['types'])):
        loss_i = losses['types'][i]
        coef_i = losses['coef'][i]
        if loss_i.__class__.__name__ == 'MixedLoss':
            mixed_loss_list = loss_i(sample_dict, net_out_dict)
            for mixed_loss in mixed_loss_list:
                loss_list.append(coef_i * mixed_loss)
        else:
            loss_list.append(coef_i * loss_i(sample_dict, net_out_dict))
    return loss_list


def train(model,
          train_dataset,
          losses,
          optimizer,
          val_dataset=None,
          save_dir='output',
          iters=10000,
          batch_size=2,
          resume_model=None,
          save_interval=1000,
          log_iters=10,
          num_workers=0,
          use_vdl=False,
          keep_checkpoint_max=5,
          postprocessor=None,
          eval_sem=False,
          eval_ins=False,
          precision='fp32',
          amp_level='O1'):
    """
    Launch training.

    Args:
        model（nn.Layer): A panoptic segmentation model.
        train_dataset (paddle.io.Dataset): Used to read and process training datasets.
        losses (dict): A dict including 'types' and 'coef'. `losses['types']` is a list of loss objects, while `losses['coef']` 
            is a list of the relevant coefficients. `len(losses['coef'])` should be equal to 1 or `len(losses['types'])`.
        optimizer (paddle.optimizer.Optimizer): The optimizer used for model training.
        val_dataset (paddle.io.Dataset, optional): Used to read and process validation datasets.
        save_dir (str, optional): The directory for saving the model snapshot. Default: 'output'.
        iters (int, optional): How may iterations to train the model for. Defualt: 10000.
        batch_size (int, optional): The mini-batch size. If multiple GPUs are used, this is the mini-batch size on each GPU.
            Default: 2.
        resume_model (str, optional): The path of the model to resume. Default: None.
        save_interval (int, optional): How many iterations to save a model snapshot once during training. Default: 1000.
        log_iters (int, optional): Display logging information at every `log_iters`. Default: 10.
        num_workers (int, optional): The number of workers used by the data loader. Default: 0.
        use_vdl (bool, optional): Whether or not to use VisualDL during training. Default: False.
        keep_checkpoint_max (int, optional): The maximum number of checkpoints to save. Default: 5.
        eval_sem (bool, optional): Whether or not to calculate semantic segmentation metrics during validation. Default: False.
        eval_ins (bool, optional): Whether or not to calculate instance segmentation metrics during validation. Default: False.
        precision (str, optional): If `precision` is 'fp16', enable automatic mixed precision training. Default: 'fp32'.
        amp_level (str, optional): The auto mixed precision level. Choices are 'O1' and 'O2'. Default: 'O1'.
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
        os.makedirs(save_dir, exist_ok=True)

    # use amp
    if precision == 'fp16':
        logger.info("Use AMP to train. AMP level = {}.".format(amp_level))
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        if amp_level == 'O2':
            model, optimizer = paddle.amp.decorate(
                models=model,
                optimizers=optimizer,
                level='O2',
                save_dtype='float32')

    if nranks > 1:
        paddle.distributed.fleet.init(is_collective=True)
        optimizer = paddle.distributed.fleet.distributed_optimizer(optimizer)
        ddp_model = paddle.distributed.fleet.distributed_model(model)

    batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True,
        worker_init_fn=worker_init_fn,
        collate_fn=train_dataset.collate
        if hasattr(train_dataset, 'collate') else None)

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
            images = data['img']
            if hasattr(model, 'data_format') and model.data_format == 'NHWC':
                images = images.transpose((0, 2, 3, 1))

            if precision == 'fp16':
                with paddle.amp.auto_cast(
                        level=amp_level,
                        enable=True,
                        custom_white_list={
                            'elementwise_add', 'batch_norm', 'sync_batch_norm'
                        },
                        custom_black_list={'bilinear_interp_v2'}):
                    net_out = ddp_model(images) if nranks > 1 else model(images)
                    loss_list = loss_computation(data, net_out, losses=losses)
                    loss = sum(loss_list)

                scaled = scaler.scale(loss)  # Scale the loss
                scaled.backward()  # Do backward
                if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                    scaler.minimize(optimizer.user_defined_optimizer, scaled)
                else:
                    scaler.minimize(optimizer, scaled)  # Update parameters
            else:
                net_out = ddp_model(images) if nranks > 1 else model(images)
                loss_list = loss_computation(data, net_out, losses=losses)
                loss = sum(loss_list)
                loss.backward()

            lr = optimizer.get_lr()

            # Update lr.
            if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                lr_sche = optimizer.user_defined_optimizer._learning_rate
            else:
                lr_sche = optimizer._learning_rate
            if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
                if isinstance(lr_sche, paddle.optimizer.lr.ReduceOnPlateau):
                    lr_sche.step(loss)
                else:
                    lr_sche.step()

            model.clear_gradients()
            avg_loss += float(loss)
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
                    .format((iter - 1
                             ) // iters_per_epoch + 1, iter, iters, avg_loss,
                            lr, avg_train_batch_cost, avg_train_reader_cost,
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

            # Save model
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

            # Eval model
            if (iter % save_interval == 0 or iter == iters) and (
                    val_dataset is not None) and local_rank == 0:
                num_workers = 1 if num_workers > 0 else 0
                results = evaluate(
                    model,
                    val_dataset,
                    postprocessor,
                    num_workers=num_workers,
                    print_detail=False,
                    eval_sem=eval_sem,
                    eval_ins=eval_ins,
                    precision=precision,
                    amp_level=amp_level)

                pq = results['pan_metrics']['All']['pq']
                desc = "[EVAL] PQ: {:.4f}".format(pq)
                if eval_sem:
                    miou = results['sem_metrics']['mIoU']
                    desc += ", mIoU: {:.4f}".format(miou)
                if eval_ins:
                    map = results['ins_metrics']['mAP']
                    desc += ", mAP: {:.4f}".format(map)
                    map50 = results['ins_metrics']['mAP50']
                    desc += ", mAP50: {:.4f}".format(map50)
                logger.info(desc)
                model.train()

            # Save best model and add evaluation results to vdl.
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
                        '[EVAL] The model with the best validation PQ ({:.4f}) was saved at iter {}.'
                        .format(best_pq, best_model_iter))

                    if use_vdl:
                        log_writer.add_scalar('Evaluate/PQ', pq, iter)
                        if eval_sem:
                            log_writer.add_scalar('Evaluate/mIoU', miou, iter)
                        if eval_ins:
                            log_writer.add_scalar('Evaluate/mAP', map, iter)
                            log_writer.add_scalar('Evaluate/mAP50', map50, iter)
            batch_start = time.time()

    # Calculate FLOPs
    if local_rank == 0 and not (precision == 'fp16' and amp_level == 'O2'):
        _, c, h, w = data['img'].shape
        flops = paddle.flops(
            model, [1, c, h, w],
            custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})

    # Sleep for a second to let dataloader release resources.
    time.sleep(1)
    if use_vdl:
        log_writer.close()

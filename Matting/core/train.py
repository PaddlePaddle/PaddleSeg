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
from collections import deque, defaultdict
import shutil

import numpy as np
import paddle
import paddle.nn.functional as F
from paddleseg.utils import TimeAverager, calculate_eta, resume, logger

from core.val import evaluate


def visual_in_traning(log_writer, vis_dict, step):
    """
    Visual in vdl

    Args:
        log_writer (LogWriter): The log writer of vdl.
        vis_dict (dict): Dict of tensor. The shape of thesor is (C, H, W)
    """
    for key, value in vis_dict.items():
        value_shape = value.shape
        if value_shape[0] not in [1, 3]:
            value = value[0]
            value = value.unsqueeze(0)
        value = paddle.transpose(value, (1, 2, 0))
        min_v = paddle.min(value)
        max_v = paddle.max(value)
        if (min_v > 0) and (max_v < 1):
            value = value * 255
        elif (min_v < 0 and min_v >= -1) and (max_v <= 1):
            value = (1 + value) / 2 * 255
        else:
            value = (value - min_v) / (max_v - min_v) * 255

        value = value.astype('uint8')
        value = value.numpy()
        log_writer.add_image(tag=key, img=value, step=step)


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
          log_image_iters=1000,
          num_workers=0,
          use_vdl=False,
          losses=None,
          keep_checkpoint_max=5,
          eval_begin_iters=None):
    """
    Launch training.
    Args:
        model（nn.Layer): A matting model.
        train_dataset (paddle.io.Dataset): Used to read and process training datasets.
        val_dataset (paddle.io.Dataset, optional): Used to read and process validation datasets.
        optimizer (paddle.optimizer.Optimizer): The optimizer.
        save_dir (str, optional): The directory for saving the model snapshot. Default: 'output'.
        iters (int, optional): How may iters to train the model. Defualt: 10000.
        batch_size (int, optional): Mini batch size of one gpu or cpu. Default: 2.
        resume_model (str, optional): The path of resume model.
        save_interval (int, optional): How many iters to save a model snapshot once during training. Default: 1000.
        log_iters (int, optional): Display logging information at every log_iters. Default: 10.
        log_image_iters (int, optional): Log image to vdl. Default: 1000.
        num_workers (int, optional): Num workers for data loader. Default: 0.
        use_vdl (bool, optional): Whether to record the data to VisualDL during training. Default: False.
        losses (dict, optional): A dict of loss, refer to the loss function of the model for details. Default: None.
        keep_checkpoint_max (int, optional): Maximum number of checkpoints to save. Default: 5.
        eval_begin_iters (int): The iters begin evaluation. It will evaluate at iters/2 if it is None. Defalust: None.
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

    avg_loss = defaultdict(float)
    iters_per_epoch = len(batch_sampler)
    best_sad = np.inf
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

            # model input
            if nranks > 1:
                logit_dict = ddp_model(data)
            else:
                logit_dict = model(data)
            loss_dict = model.loss(logit_dict, data, losses)

            loss_dict['all'].backward()

            optimizer.step()
            lr = optimizer.get_lr()
            if isinstance(optimizer._learning_rate,
                          paddle.optimizer.lr.LRScheduler):
                optimizer._learning_rate.step()
            model.clear_gradients()

            for key, value in loss_dict.items():
                avg_loss[key] += value.numpy()[0]
            batch_cost_averager.record(
                time.time() - batch_start, num_samples=batch_size)

            if (iter) % log_iters == 0 and local_rank == 0:
                for key, value in avg_loss.items():
                    avg_loss[key] = value / log_iters
                remain_iters = iters - iter
                avg_train_batch_cost = batch_cost_averager.get_average()
                avg_train_reader_cost = reader_cost_averager.get_average()
                eta = calculate_eta(remain_iters, avg_train_batch_cost)
                logger.info(
                    "[TRAIN] epoch={}, iter={}/{}, loss={:.4f}, lr={:.6f}, batch_cost={:.4f}, reader_cost={:.5f}, ips={:.4f} samples/sec | ETA {}"
                    .format((iter - 1) // iters_per_epoch + 1, iter, iters,
                            avg_loss['all'], lr, avg_train_batch_cost,
                            avg_train_reader_cost,
                            batch_cost_averager.get_ips_average(), eta))
                # print loss
                loss_str = '[TRAIN] [LOSS] '
                loss_str = loss_str + 'all={:.4f}'.format(avg_loss['all'])
                for key, value in avg_loss.items():
                    if key != 'all':
                        loss_str = loss_str + ' ' + key + '={:.4f}'.format(
                            value)
                logger.info(loss_str)
                if use_vdl:
                    for key, value in avg_loss.items():
                        log_tag = 'Train/' + key
                        log_writer.add_scalar(log_tag, value, iter)

                    log_writer.add_scalar('Train/lr', lr, iter)
                    log_writer.add_scalar('Train/batch_cost',
                                          avg_train_batch_cost, iter)
                    log_writer.add_scalar('Train/reader_cost',
                                          avg_train_reader_cost, iter)
                    if iter % log_image_iters == 0:
                        vis_dict = {}
                        # ground truth
                        vis_dict['ground truth/img'] = data['img'][0]
                        for key in data['gt_fields']:
                            key = key[0]
                            vis_dict['/'.join(['ground truth',
                                               key])] = data[key][0]
                        # predict
                        for key, value in logit_dict.items():
                            vis_dict['/'.join(['predict',
                                               key])] = logit_dict[key][0]
                        visual_in_traning(
                            log_writer=log_writer, vis_dict=vis_dict, step=iter)

                for key in avg_loss.keys():
                    avg_loss[key] = 0.
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
            if eval_begin_iters is None:
                eval_begin_iters = iters // 2
            if (iter % save_interval == 0 or iter == iters) and (
                    val_dataset is
                    not None) and local_rank == 0 and iter >= eval_begin_iters:
                num_workers = 1 if num_workers > 0 else 0
                sad, mse = evaluate(
                    model,
                    val_dataset,
                    num_workers=0,
                    print_detail=True,
                    save_results=False)
                model.train()

            # save best model and add evaluation results to vdl
            if (iter % save_interval == 0 or iter == iters) and local_rank == 0:
                if val_dataset is not None and iter >= eval_begin_iters:
                    if sad < best_sad:
                        best_sad = sad
                        best_model_iter = iter
                        best_model_dir = os.path.join(save_dir, "best_model")
                        paddle.save(
                            model.state_dict(),
                            os.path.join(best_model_dir, 'model.pdparams'))
                    logger.info(
                        '[EVAL] The model with the best validation sad ({:.4f}) was saved at iter {}.'
                        .format(best_sad, best_model_iter))

                    if use_vdl:
                        log_writer.add_scalar('Evaluate/SAD', sad, iter)
                        log_writer.add_scalar('Evaluate/MSE', mse, iter)

            batch_start = time.time()

    # Sleep for half a second to let dataloader release resources.
    time.sleep(0.5)
    if use_vdl:
        log_writer.close()

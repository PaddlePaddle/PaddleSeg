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
import pickle
import shutil

import numpy as np
import paddle
import paddle.nn.functional as F
from paddleseg.utils import TimeAverager, calculate_eta, resume, logger

from .val import evaluate


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


def save_best(best_model_dir, metrics_data, iter):
    with open(os.path.join(best_model_dir, 'best_metrics.txt'), 'w') as f:
        for key, value in metrics_data.items():
            line = key + ' ' + str(value) + '\n'
            f.write(line)
        f.write('iter' + ' ' + str(iter) + '\n')


def get_best(best_file, metrics, resume_model=None):
    '''Get best metrics and iter from file'''
    best_metrics_data = {}
    if os.path.exists(best_file) and (resume_model is not None):
        values = []
        with open(best_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                key, value = line.split(' ')
                best_metrics_data[key] = eval(value)
                if key == 'iter':
                    best_iter = eval(value)
    else:
        for key in metrics:
            best_metrics_data[key] = np.inf
        best_iter = -1
    return best_metrics_data, best_iter


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
          eval_begin_iters=None,
          metrics='sad'):
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
        metrics(str|list, optional): The metrics to evaluate, it may be the combination of ("sad", "mse", "grad", "conn"). 
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
        return_list=True, )

    if use_vdl:
        from visualdl import LogWriter
        log_writer = LogWriter(save_dir)

    if isinstance(metrics, str):
        metrics = [metrics]
    elif not isinstance(metrics, list):
        metrics = ['sad']
    best_metrics_data, best_iter = get_best(
        os.path.join(save_dir, 'best_model', 'best_metrics.txt'),
        metrics,
        resume_model=resume_model)
    avg_loss = defaultdict(float)
    iters_per_epoch = len(batch_sampler)
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

            logit_dict, loss_dict = ddp_model(data) if nranks > 1 else model(
                data)

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
                # loss info
                loss_str = ' ' * 26 + '\t[LOSSES]'
                loss_str = loss_str
                for key, value in avg_loss.items():
                    if key != 'all':
                        loss_str = loss_str + ' ' + key + '={:.4f}'.format(
                            value)
                logger.info(
                    "[TRAIN] epoch={}, iter={}/{}, loss={:.4f}, lr={:.6f}, batch_cost={:.4f}, reader_cost={:.5f}, ips={:.4f} samples/sec | ETA {}\n{}\n"
                    .format((iter - 1) // iters_per_epoch + 1, iter, iters,
                            avg_loss['all'], lr, avg_train_batch_cost,
                            avg_train_reader_cost,
                            batch_cost_averager.get_ips_average(
                            ), eta, loss_str))
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
                            vis_dict['/'.join(['ground truth', key])] = data[
                                key][0]
                        # predict
                        for key, value in logit_dict.items():
                            vis_dict['/'.join(['predict', key])] = logit_dict[
                                key][0]
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
                    val_dataset is not None
            ) and local_rank == 0 and iter >= eval_begin_iters:
                num_workers = 1 if num_workers > 0 else 0
                metrics_data = evaluate(
                    model,
                    val_dataset,
                    num_workers=1,
                    print_detail=True,
                    save_results=False,
                    metrics=metrics)
                model.train()

            # save best model and add evaluation results to vdl
            if (iter % save_interval == 0 or iter == iters) and local_rank == 0:
                if val_dataset is not None and iter >= eval_begin_iters:
                    if metrics_data[metrics[0]] < best_metrics_data[metrics[0]]:
                        best_iter = iter
                        best_metrics_data = metrics_data.copy()
                        best_model_dir = os.path.join(save_dir, "best_model")
                        paddle.save(
                            model.state_dict(),
                            os.path.join(best_model_dir, 'model.pdparams'))
                        save_best(best_model_dir, best_metrics_data, iter)

                    show_list = []
                    for key, value in best_metrics_data.items():
                        show_list.append((key, value))
                    log_str = '[EVAL] The model with the best validation {} ({:.4f}) was saved at iter {}. while'.format(
                        show_list[0][0], show_list[0][1], best_iter)
                    for i in range(1, len(show_list)):
                        log_str = log_str + ' {}: {:.4f},'.format(
                            show_list[i][0], show_list[i][1])
                    log_str = log_str[:-1]
                    logger.info(log_str)

                    if use_vdl:
                        for key, value in metrics_data.items():
                            log_writer.add_scalar('Evaluate/' + key, value,
                                                  iter)

            batch_start = time.time()

    # Sleep for half a second to let dataloader release resources.
    time.sleep(0.5)
    if use_vdl:
        log_writer.close()

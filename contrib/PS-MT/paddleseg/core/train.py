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
from itertools import cycle

import paddle
import paddle.nn.functional as F

from paddleseg.utils import (TimeAverager, calculate_eta, resume, logger,
                             worker_init_fn, train_profiler, op_flops_funs)
from paddleseg.core.val import evaluate
from paddleseg.core import train_assit


def check_logits_losses(logits_list, losses):
    len_logits = len(logits_list)
    len_losses = len(losses['types'])
    if len_logits != len_losses:
        raise RuntimeError(
            'The length of logits_list should equal to the types of loss config: {} != {}.'
                .format(len_logits, len_losses))


def loss_computation(logits_list, labels, edges, losses):
    check_logits_losses(logits_list, losses)
    loss_list = []
    for i in range(len(logits_list)):
        logits = logits_list[i]
        loss_i = losses['types'][i]
        coef_i = losses['coef'][i]
        if loss_i.__class__.__name__ in ('BCELoss',) and loss_i.edge_label:
            # Use edges as labels According to loss type.
            loss_list.append(coef_i * loss_i(logits, edges))
        elif loss_i.__class__.__name__ == 'MixedLoss':
            mixed_loss_list = loss_i(logits, labels)
            for mixed_loss in mixed_loss_list:
                loss_list.append(coef_i * mixed_loss)
        elif loss_i.__class__.__name__ in ("KLLoss",):
            loss_list.append(coef_i *
                             loss_i(logits_list[0], logits_list[1].detach()))
        else:
            loss_list.append(coef_i * loss_i(logits, labels))
    return loss_list


def train(model,
          train_sup_dataset,
          train_unsup_dataset,
          val_dataset=None,
          optimizer_t1=None,
          optimizer_t2=None,
          optimizer_s=None,
          lr_scheduler_s=None,
          warm_up_num=None,
          gamma=0.5,
          cons_w_unsup=0.5,
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
          test_config=None,
          precision='fp32',
          amp_level='O1',
          profiler_options=None,
          to_static_training=False):
    """
    Launch training.

    Args:
        cons_w_unsup(float, optional): consistency weight of unsupervised predict
        gamma (float, optional): weight of two teacher predict .Default: 0.5.
        model: （nn.Layer): A semantic segmentation model.
        optimizer_t1: The optimizer.

        train_sup_dataset (paddle.io.Dataset): Used to read and process training datasets.
        val_dataset (paddle.io.Dataset, optional): Used to read and process validation datasets.
        optimizer (paddle.optimizer.Optimizer): The optimizer.
        save_dir (str, optional): The directory for saving the model snapshot. Default: 'output'.
        iters (int, optional): How may iters to train the model. Default: 10000.
        batch_size (int, optional): Mini batch size of one gpu or cpu. Default: 2.
        resume_model (str, optional): The path of resume model.
        save_interval (int, optional): How many iters to save a model snapshot once during training. Default: 1000.
        log_iters (int, optional): Display logging information at every log_iters. Default: 10.
        num_workers (int, optional): Num workers for data loader. Default: 0.
        use_vdl (bool, optional): Whether to record the data to VisualDL during training. Default: False.
        losses (dict, optional): A dict including 'types' and 'coef'. The length of coef should equal to 1 or len(losses['types']).
            The 'types' item is a list of object of paddleseg.models.losses while the 'coef' item is a list of the relevant coefficient.
        keep_checkpoint_max (int, optional): Maximum number of checkpoints to save. Default: 5.
        test_config(dict, optional): Evaluation config.
        precision (str, optional): Use AMP if precision='fp16'. If precision='fp32', the training is normal.
        amp_level (str, optional): Auto mixed precision level. Accepted values are “O1” and “O2”: O1 represent mixed precision, 
            the input data type of each operator will be casted by white_list and black_list; O2 represent Pure fp16, all operators 
            parameters and input data will be casted to fp16, except operators in black_list, don’t support fp16 kernel and batchnorm. Default is O1(amp)
        profiler_options (str, optional): The option of train profiler.
        to_static_training (bool, optional): Whether to use @to_static for training.
    """
    # 有监督数据集
    sup_batch_sampler = paddle.io.DistributedBatchSampler(
        train_sup_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    sup_loader = paddle.io.DataLoader(
        train_sup_dataset,
        batch_sampler=sup_batch_sampler,
        num_workers=0,
        return_list=True,
        worker_init_fn=worker_init_fn,
    )
    # 无监督数据集
    unsup_batch_sampler = paddle.io.DistributedBatchSampler(
        train_unsup_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    unsup_loader = paddle.io.DataLoader(
        train_unsup_dataset,
        batch_sampler=unsup_batch_sampler,
        num_workers=0,
        return_list=True,
        worker_init_fn=worker_init_fn,
    )
    
    # 预热训练
    if warm_up_num is not None:
        for num in range(0, warm_up_num):
            train_assit.warm_up(model, sup_loader, optimizer1=optimizer_t1, optimizer2=optimizer_t2,
                                optimizer_s=optimizer_s, id=1)
            train_assit.warm_up(model, sup_loader, optimizer1=optimizer_t1, optimizer2=optimizer_t2,
                                optimizer_s=optimizer_s, id=2)
            train_assit.warm_up(model, sup_loader, optimizer1=optimizer_t1, optimizer2=optimizer_t2,
                                optimizer_s=optimizer_s, id=3)
            if num == warm_up_num - 1:
                del optimizer_t1
                del optimizer_t2
    model.freeze_teachers_parameters()
    model.train()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank

    start_iter = 0

    if not os.path.isdir(save_dir):
        if os.path.exists(save_dir):
            os.remove(save_dir)
        os.makedirs(save_dir, exist_ok=True)
    ddp=False  # 分布式训练
    if nranks > 1:
        ddp = True
        paddle.distributed.fleet.init(is_collective=True)
        optimizer1 = paddle.distributed.fleet.distributed_optimizer(
            optimizer_t1)  # The return is Fleet object
        optimizer2 = paddle.distributed.fleet.distributed_optimizer(
            optimizer_t2)  # The return is Fleet object
        optimizers = paddle.distributed.fleet.distributed_optimizer(
            optimizer_s)  # The return is Fleet object
        ddp_model = paddle.distributed.fleet.distributed_model(model)

    # use amp
    if precision == 'fp16':
        logger.info('use amp to train')
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

    if use_vdl:
        from visualdl import LogWriter
        log_writer = LogWriter(save_dir)

    if to_static_training:
        model = paddle.jit.to_static(model)
        logger.info("Successfully applied @to_static")

    avg_loss = 0.0
    avg_loss_list = []
    iters_per_epoch = len(unsup_loader)
    best_mean_iou = -1.0
    best_model_iter = -1
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    save_models = deque()
    batch_start = time.time()
    iter = start_iter
    while iter < iters:
        dataloader = (zip(cycle(sup_loader), unsup_loader))
        for data in dataloader:
            iter += 1
            if iter > iters:
                version = paddle.__version__
                if version == '2.1.2':
                    continue
                else:
                    break
            reader_cost_averager.record(time.time() - batch_start)

            if model.mode == "semi":
                input_l, target_l = data[0]['strong_aug'], data[0]['label']
                (input_ul_wk, input_ul_str, target_ul) = data[1]['weak_aug'], data[1]['strong_aug'], data[1]['label']

                input_ul_wk, input_ul_str, target_ul = input_ul_wk.cuda(blocking=True), \
                                                       input_ul_str.cuda(blocking=True), \
                                                       target_ul.cuda(blocking=True).astype('int64')
            else:
                input_l, target_l = data[0]['strong_aug'], data[0]['label']
                input_ul_wk, input_ul_str, target_ul = None, None, None
            # strong aug for all the supervised images
            input_l, target_l = input_l.cuda(blocking=False), target_l.cuda(blocking=False).astype('int64')
            curr_id = 1 if ((iter - 1) // iters_per_epoch+1)%2!=0 else 2
            if model.mode == "semi":
                t1_prob, t2_prob = train_assit.predict_with_out_grad(model, input_ul_wk)
                # calculate the assistance result from other teacher
                if curr_id == 1:
                    t2_prob = train_assit.assist_mask_calculate(core_predict=t1_prob,
                                                                assist_predict=t2_prob,
                                                                topk=7)
                else:
                    t1_prob = train_assit.assist_mask_calculate(core_predict=t2_prob,
                                                                assist_predict=t1_prob,
                                                                topk=7)
                predict_target_ul = gamma * t1_prob + (1 - gamma) * t2_prob
            else:
                predict_target_ul = None

            if ddp:
                paddle.distributed.barrier()
            input_l, target_l, input_ul_str, predict_target_ul = train_assit.cut_mix(input_l, target_l,
                                                                                     input_ul_str,
                                                                                     predict_target_ul)
            edges = None

            if nranks > 1:
                output_l, output_ul = ddp_model(x_l=input_l, x_ul=input_ul_str, id=0, warm_up=False)
            else:
                output_l, output_ul = model(x_l=input_l, x_ul=input_ul_str, id=0, warm_up=False)
            # 有监督损失
            loss_sup = F.cross_entropy(output_l, target_l, ignore_index=255, axis=1)
            logits_lists=[output_ul]
            loss_list = loss_computation(
                logits_list=logits_lists,
                labels=predict_target_ul,
                edges=edges,
                losses=losses)

            consist_weight = cons_w_unsup(epoch=(iter - 1) // iters_per_epoch, curr_iter=iter)
            # loss_list[0][0]为无监督损失
            loss = loss_sup + loss_list[0][0] * consist_weight
            loss = loss.mean()
            loss.backward()
            # if the optimizer is ReduceOnPlateau, the loss is the one which has been pass into step.
            if isinstance(optimizer_s, paddle.optimizer.lr.ReduceOnPlateau):
                optimizer_s.step(loss)
            else:
                optimizer_s.step()
                optimizer_s.clear_grad()

            # update lr
            lr_scheduler_s.step(epoch=(iter - 1) // iters_per_epoch)
            lr = lr_scheduler_s.get_lr()
            # model.clear_gradients()
            train_profiler.add_profiler_step(profiler_options)
            with paddle.no_grad():
                if curr_id == 1:
                    train_assit.update_teachers(model,teacher_encoder=model.encoder1,
                                         teacher_decoder=model.decoder1)
                else:
                    train_assit.update_teachers(model,teacher_encoder=model.encoder2,
                                         teacher_decoder=model.decoder2)
                if ddp:
                    paddle.distributed.barrier()

            # model.clear_gradients()
            avg_loss += float(loss)
            batch_cost_averager.record(
                time.time() - batch_start, num_samples=batch_size)

            if (iter) % log_iters == 0 and local_rank == 0:
                avg_loss /= log_iters
                avg_loss_list = avg_loss
                remain_iters = iters - iter
                avg_train_batch_cost = batch_cost_averager.get_average()
                avg_train_reader_cost = reader_cost_averager.get_average()
                eta = calculate_eta(remain_iters, avg_train_batch_cost)
                logger.info(
                    "[TRAIN] epoch: {}, iter: {}/{}, loss: {:.4f}, lr: {:.6f}, batch_cost: {:.4f}, reader_cost: {:.5f}, ips: {:.4f} samples/sec | ETA {}"
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

            if (iter % save_interval == 0 or
                iter == iters) and (val_dataset is not None):
                num_workers = 1 if num_workers > 0 else 0

                if test_config is None:
                    test_config = {}

                acc, Iou, mean_iou = evaluate(
                    model,
                    val_dataset,
                    aug_eval=True,
                    num_workers=num_workers,
                    precision=precision,
                    amp_level=amp_level,
                    **test_config)
                model.freeze_teachers_parameters()
                model.train()

            if (iter % save_interval == 0 or iter == iters) and local_rank == 0:
                current_save_dir = os.path.join(save_dir,
                                                "iter_{}".format(iter))
                if not os.path.isdir(current_save_dir):
                    os.makedirs(current_save_dir)
                paddle.save(model.state_dict(),
                            os.path.join(current_save_dir, 'model.pdparams'))
                paddle.save(optimizer_s.state_dict(),
                            os.path.join(current_save_dir, 'model.pdopt'))
                save_models.append(current_save_dir)
                if len(save_models) > keep_checkpoint_max > 0:
                    model_to_remove = save_models.popleft()
                    shutil.rmtree(model_to_remove)

                if val_dataset is not None:
                    if mean_iou > best_mean_iou:
                        best_mean_iou = mean_iou
                        best_model_iter = iter
                        best_model_dir = os.path.join(save_dir, "best_model")
                        paddle.save(
                            model.state_dict(),
                            os.path.join(best_model_dir, 'model.pdparams'))
                    logger.info(
                        '[EVAL] The model with the best validation mIoU ({:.4f}) was saved at iter {}.'
                            .format(best_mean_iou, best_model_iter))

                    if use_vdl:
                        log_writer.add_scalar('Evaluate/mIoU', mean_iou, iter)
                        log_writer.add_scalar('Evaluate/Acc', acc, iter)
            batch_start = time.time()

    # Calculate flops.
    if local_rank == 0 and not (precision == 'fp16' and amp_level == 'O2'):
        _, c, h, w = input_ul_str.shape
        _ = paddle.flops(
            model, [1, c, h, w],
            custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})

    # Sleep for a second to let dataloader release resources.
    time.sleep(1)
    if use_vdl:
        log_writer.close()

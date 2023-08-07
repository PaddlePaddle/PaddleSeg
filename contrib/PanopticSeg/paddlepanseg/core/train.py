# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
                             worker_init_fn, train_profiler, op_flops_funs)

from paddlepanseg.core.val import evaluate
from paddlepanseg.core.runner import PanSegRunner
from paddlepanseg.core.launcher import AMPLauncher


class _DistOptimizerWrapper(object):
    def __init__(self, dist_optim):
        assert isinstance(dist_optim, paddle.distributed.fleet.Fleet)
        self._optim = dist_optim

    def __getattr__(self, name):
        # XXX: We choose to rewrite `__getattr__` here because it is simpler.
        # However, this may cause performance drop.
        # First try to find the attribute in `self._optim`
        try:
            return getattr(self._optim, name)
        except AttributeError:
            # If an attribute is not found in `self._optim`, search it in the internal optimizer
            real_optim = self._optim.user_defined_optimizer
            if not hasattr(real_optim, name):
                raise AttributeError
            else:
                return getattr(real_optim, name)


def train(model,
          train_dataset,
          losses,
          optimizer,
          postprocessor,
          runner,
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
          eval_sem=False,
          eval_ins=False,
          precision='fp32',
          amp_level='O1',
          profiler_options=None,
          to_static_training=False):
    """
    Launch training.

    Args:
        model（nn.Layer): A panoptic segmentation model.
        train_dataset (paddle.io.Dataset): Used to read and process training datasets.
        losses (dict): A dict including 'types' and 'coef'. `losses['types']` is a list of loss objects, while `losses['coef']` 
            is a list of the relevant coefficients. `len(losses['coef'])` should be equal to 1 or `len(losses['types'])`.
        optimizer (paddle.optimizer.Optimizer): The optimizer used for model training.
        postprocessor (paddlepanseg.postprocessors.Postprocessor): Used to postprocess model output in model validation.
        runner (paddlepanseg.core.runners.PanSegRunner): Used to define how the components interact.
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
        profiler_options (str, optional): Options of the training profiler.
        to_static_training (bool, optional): Whether or not to apply dynamic-to-static model training.
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

    # Use AMP
    if precision == 'fp16':
        logger.info("Use AMP to train. AMP level = {}.".format(amp_level))

    if nranks > 1:
        paddle.distributed.fleet.init(is_collective=True)
        optimizer = paddle.distributed.fleet.distributed_optimizer(optimizer)
        if isinstance(optimizer, paddle.distributed.fleet.Fleet):
            optimizer = _DistOptimizerWrapper(optimizer)
        ddp_model = paddle.distributed.fleet.distributed_model(model)

    # Build batch sampler and data loader
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

    if to_static_training:
        model = paddle.jit.to_static(model)
        logger.info("Successfully applied `paddle.jit.to_static`")

    # Bind components to runner
    if nranks > 1:
        runner.bind(model=ddp_model, criteria=losses, optimizer=optimizer)
    else:
        runner.bind(model=model, criteria=losses, optimizer=optimizer)

    # Create launcher
    launcher = AMPLauncher(
        runner=runner, precision=precision, amp_level=amp_level)

    if use_vdl:
        # Build log writer
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

    # By default we adopt iteration-based training
    iter = start_iter
    while iter < iters:
        for data in loader:
            iter += 1
            if iter > iters:
                break

            reader_cost_averager.record(time.time() - batch_start)

            loss, loss_list = launcher.train_step(
                data=data, return_loss_list=True)

            train_profiler.add_profiler_step(profiler_options)

            avg_loss += float(loss)
            if not avg_loss_list:
                avg_loss_list = [l.numpy() for l in loss_list]
            else:
                for i in range(len(loss_list)):
                    avg_loss_list[i] += loss_list[i].numpy()

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=batch_size)

            if (iter) % log_iters == 0 and local_rank == 0:
                lr = launcher.runner.optimizer.get_lr()
                avg_loss /= log_iters
                avg_loss_list = [l.item() / log_iters for l in avg_loss_list]
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
                    amp_level=amp_level,
                    runner=runner)

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

    # Sleep for a second to let dataloader release resources.
    time.sleep(1)
    if use_vdl:
        log_writer.close()

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

import paddle.fluid as fluid
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.fluid.io import DataLoader
from paddle.incubate.hapi.distributed import DistributedBatchSampler

import utils.logging as logging
from utils import load_pretrained_model
from utils import resume
from utils import Timer, calculate_eta
from .val import evaluate


def train(model,
          train_dataset,
          places=None,
          eval_dataset=None,
          optimizer=None,
          save_dir='output',
          num_epochs=100,
          batch_size=2,
          pretrained_model=None,
          resume_model=None,
          save_interval_epochs=1,
          log_steps=10,
          num_classes=None,
          num_workers=8,
          use_vdl=False):
    ignore_index = model.ignore_index
    nranks = ParallelEnv().nranks

    start_epoch = 0
    if resume_model is not None:
        start_epoch = resume(model, optimizer, resume_model)
    elif pretrained_model is not None:
        load_pretrained_model(model, pretrained_model)

    if not os.path.isdir(save_dir):
        if os.path.exists(save_dir):
            os.remove(save_dir)
        os.makedirs(save_dir)

    if nranks > 1:
        strategy = fluid.dygraph.prepare_context()
        ddp_model = fluid.dygraph.DataParallel(model, strategy)

    batch_sampler = DistributedBatchSampler(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        places=places,
        num_workers=num_workers,
        return_list=True,
    )

    if use_vdl:
        from visualdl import LogWriter
        log_writer = LogWriter(save_dir)

    timer = Timer()
    avg_loss = 0.0
    steps_per_epoch = len(batch_sampler)
    total_steps = steps_per_epoch * (num_epochs - start_epoch)
    num_steps = 0
    best_mean_iou = -1.0
    best_model_epoch = -1
    train_reader_cost = 0.0
    train_batch_cost = 0.0
    for epoch in range(start_epoch, num_epochs):
        timer.start()
        for step, data in enumerate(loader):
            train_reader_cost += timer.elapsed_time()
            images = data[0]
            labels = data[1].astype('int64')
            if nranks > 1:
                loss = ddp_model(images, labels)
                # apply_collective_grads sum grads over multiple gpus.
                loss = ddp_model.scale_loss(loss)
                loss.backward()
                ddp_model.apply_collective_grads()
            else:
                loss = model(images, labels)
                loss.backward()
            optimizer.minimize(loss)
            model.clear_gradients()
            avg_loss += loss.numpy()[0]
            lr = optimizer.current_step_lr()
            num_steps += 1
            train_batch_cost += timer.elapsed_time()
            if num_steps % log_steps == 0 and ParallelEnv().local_rank == 0:
                avg_loss /= log_steps
                avg_train_reader_cost = train_reader_cost / log_steps
                avg_train_batch_cost = train_batch_cost / log_steps
                train_reader_cost = 0.0
                train_batch_cost = 0.0
                remain_steps = total_steps - num_steps
                eta = calculate_eta(remain_steps, avg_train_batch_cost)
                logging.info(
                    "[TRAIN] Epoch={}/{}, Step={}/{}, loss={:.4f}, lr={:.6f}, batch_cost={:.4f}, reader_cost={:.4f} | ETA {}"
                    .format(epoch + 1, num_epochs, step + 1, steps_per_epoch,
                            avg_loss * nranks, lr, avg_train_batch_cost,
                            avg_train_reader_cost, eta))
                if use_vdl:
                    log_writer.add_scalar('Train/loss', avg_loss, num_steps)
                    log_writer.add_scalar('Train/lr', lr, num_steps)
                    log_writer.add_scalar('Train/batch_cost',
                                          avg_train_batch_cost, num_steps)
                    log_writer.add_scalar('Train/reader_cost',
                                          avg_train_reader_cost, num_steps)
                avg_loss = 0.0
            timer.restart()

        if ((epoch + 1) % save_interval_epochs == 0
                or epoch + 1 == num_epochs) and ParallelEnv().local_rank == 0:
            current_save_dir = os.path.join(save_dir,
                                            "epoch_{}".format(epoch + 1))
            if not os.path.isdir(current_save_dir):
                os.makedirs(current_save_dir)
            fluid.save_dygraph(model.state_dict(),
                               os.path.join(current_save_dir, 'model'))
            fluid.save_dygraph(optimizer.state_dict(),
                               os.path.join(current_save_dir, 'model'))

            if eval_dataset is not None:
                mean_iou, avg_acc = evaluate(
                    model,
                    eval_dataset,
                    model_dir=current_save_dir,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    epoch_id=epoch + 1)
                if mean_iou > best_mean_iou:
                    best_mean_iou = mean_iou
                    best_model_epoch = epoch + 1
                    best_model_dir = os.path.join(save_dir, "best_model")
                    fluid.save_dygraph(model.state_dict(),
                                       os.path.join(best_model_dir, 'model'))
                logging.info(
                    'Current evaluated best model in eval_dataset is epoch_{}, miou={:4f}'
                    .format(best_model_epoch, best_mean_iou))

                if use_vdl:
                    log_writer.add_scalar('Evaluate/mIoU', mean_iou, epoch + 1)
                    log_writer.add_scalar('Evaluate/aAcc', avg_acc, epoch + 1)
                model.train()
    if use_vdl:
        log_writer.close()

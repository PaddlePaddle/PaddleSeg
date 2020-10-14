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

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.fluid.io import DataLoader
from paddle.io import DistributedBatchSampler
import paddle.nn.functional as F

import paddleseg.utils.logger as logger
from paddleseg.utils import load_pretrained_model
from paddleseg.utils import resume
from paddleseg.utils import Timer, calculate_eta
from paddleseg.core.val import evaluate
from paddleseg.cvlibs import callbacks


def check_logits_losses(logits, losses):
    len_logits = len(logits)
    len_losses = len(losses['types'])
    if len_logits != len_losses:
        raise RuntimeError(
            'The length of logits should equal to the types of loss config: {} != {}.'
            .format(len_logits, len_losses))


def loss_computation(logits, label, losses):
    check_logits_losses(logits, losses)
    loss = 0
    for i in range(len(logits)):
        logit = logits[i]
        if logit.shape[-2:] != label.shape[-2:]:
            logit = F.resize_bilinear(logit, label.shape[-2:])
        loss_i = losses['types'][i](logit, label)
        loss += losses['coef'][i] * loss_i
    return loss


def seg_train(model,
              train_dataset,
              places=None,
              val_dataset=None,
              losses=None,
              optimizer=None,
              save_dir='output',
              iters=10000,
              batch_size=2,
              resume_model=None,
              save_interval=1000,
              log_iters=10,
              num_workers=8):

    nranks = ParallelEnv().nranks

    start_iter = 0
    if resume_model is not None:
        start_iter = resume(model, optimizer, resume_model)

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

    out_labels = ["loss", "reader_cost", "batch_cost"]
    base_logger = callbacks.BaseLogger(period=log_iters)
    train_logger = callbacks.TrainLogger(log_freq=log_iters)
    model_ckpt = callbacks.ModelCheckpoint(
        save_dir, save_params_only=False, period=save_interval)
    vdl = callbacks.VisualDL(log_dir=os.path.join(save_dir, "log"))
    cbks_list = [base_logger, train_logger, model_ckpt, vdl]

    cbks = callbacks.CallbackList(cbks_list)
    cbks.set_model(model)
    cbks.set_optimizer(optimizer)
    cbks.set_params({
        "batch_size": batch_size,
        "total_iters": iters,
        "log_iters": log_iters,
        "verbose": 1,
        "do_validation": True,
        "metrics": out_labels,
        "iters_per_epoch": len(batch_sampler)
    })

    logs = {}
    logs = {key: 0.0 for key in out_labels}

    timer = Timer()
    timer.start()

    ############## 1 ################
    cbks.on_train_begin(logs)
    #################################

    iter = start_iter
    while iter < iters:
        for data in loader:
            iter += 1
            if iter > iters:
                break

            logs["reader_cost"] = timer.elapsed_time()
            ############## 2 ################
            cbks.on_iter_begin(iter, logs)
            #################################

            images = data[0]
            labels = data[1].astype('int64')

            if nranks > 1:
                logits = ddp_model(images)
                loss = loss_computation(logits, labels, losses)
                # apply_collective_grads sum grads over multiple gpus.
                loss = ddp_model.scale_loss(loss)
                loss.backward()
                ddp_model.apply_collective_grads()

            else:
                logits = model(images)
                loss = loss_computation(logits, labels, losses)
                loss.backward()

            optimizer.step()
            optimizer._learning_rate.step()

            model.clear_gradients()

            logs['loss'] = loss.numpy()[0]

            logs["batch_cost"] = timer.elapsed_time()

            ############## 3 ################
            cbks.on_iter_end(iter, logs)
            #################################

            timer.restart()

############### 4 ###############
    cbks.on_train_end(logs)


#################################

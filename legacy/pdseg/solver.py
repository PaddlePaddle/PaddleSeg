# coding: utf8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import sys

import paddle
import numpy as np
from utils.config import cfg
from paddle.static.amp import decorate, AutoMixedPrecisionLists


class Solver(object):
    def __init__(self, main_prog, start_prog):
        total_images = cfg.DATASET.TRAIN_TOTAL_IMAGES
        self.weight_decay = cfg.SOLVER.WEIGHT_DECAY
        self.momentum = cfg.SOLVER.MOMENTUM
        self.momentum2 = cfg.SOLVER.MOMENTUM2
        self.step_per_epoch = total_images // cfg.BATCH_SIZE
        if total_images % cfg.BATCH_SIZE != 0:
            self.step_per_epoch += 1
        self.total_step = cfg.SOLVER.NUM_EPOCHS * self.step_per_epoch
        self.main_prog = main_prog
        self.start_prog = start_prog
        self.warmup_step = cfg.SOLVER.LR_WARMUP_STEPS if cfg.SOLVER.LR_WARMUP else -1
        self.decay_step = self.total_step - self.warmup_step
        self.decay_epochs = cfg.SOLVER.NUM_EPOCHS - self.warmup_step / self.step_per_epoch

    def piecewise_decay(self):
        gamma = cfg.SOLVER.GAMMA
        bd = [self.step_per_epoch * e for e in cfg.SOLVER.DECAY_EPOCH]
        lr = [cfg.SOLVER.LR * (gamma**i) for i in range(len(bd) + 1)]
        decayed_lr = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=bd, values=lr)
        return decayed_lr

    def poly_decay(self):
        power = cfg.SOLVER.POWER
        decayed_lr = paddle.optimizer.lr.PolynomialDecay(
            cfg.SOLVER.LR, self.total_step, end_lr=0, power=power)
        return decayed_lr

    # There is not paddle.optimizer.lr.CosineDecay in paddle2.1 version
    # def cosine_decay(self):
    #     decayed_lr = paddle.fluid.layers.cosine_decay()
    #         cfg.SOLVER.LR, self.step_per_epoch, self.decay_epochs)
    #     return decayed_lr

    def get_lr(self, lr_policy):
        if lr_policy.lower() == 'poly':
            decayed_lr = self.poly_decay()
        elif lr_policy.lower() == 'piecewise':
            decayed_lr = self.piecewise_decay()
        # elif lr_policy.lower() == 'cosine':
        #     decayed_lr = self.cosine_decay()
        else:
            raise Exception(
                "unsupport learning decay policy! only support poly,piecewise")

        return decayed_lr

    def sgd_optimizer(self, lr_policy, loss):
        decayed_lr = self.get_lr(lr_policy)
        optimizer = paddle.optimizer.Momentum(
            learning_rate=decayed_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        if cfg.MODEL.FP16:
            print('use amp')
            custom_black_list = {}
            amp_lists = AutoMixedPrecisionLists(
                custom_black_list=custom_black_list)
            assert isinstance(cfg.MODEL.SCALE_LOSS, float) or isinstance(cfg.MODEL.SCALE_LOSS, str), \
                "data type of MODEL.SCALE_LOSS must be float or str"
            if isinstance(cfg.MODEL.SCALE_LOSS, float):
                optimizer = decorate(
                    optimizer,
                    amp_lists=amp_lists,
                    init_loss_scaling=cfg.MODEL.SCALE_LOSS,
                    use_dynamic_loss_scaling=False)
            else:
                assert cfg.MODEL.SCALE_LOSS.lower() in [
                    'dynamic'
                ], "if MODEL.SCALE_LOSS is a string,\
                 must be set as 'DYNAMIC'!"

                optimizer = decorate(
                    optimizer,
                    amp_lists=amp_lists,
                    use_dynamic_loss_scaling=True)

        # optimizer.minimize(loss)
        return decayed_lr, optimizer

    def adam_optimizer(self, lr_policy, loss):
        decayed_lr = self.get_lr(lr_policy)
        optimizer = paddle.optimizer.Adam(
            learning_rate=decayed_lr,
            beta1=self.momentum,
            beta2=self.momentum2,
            weight_decay=self.weight_decay,
        )
        #optimizer.minimize(loss)
        return decayed_lr, optimizer

    def optimise(self, loss):
        lr_policy = cfg.SOLVER.LR_POLICY
        opt = cfg.SOLVER.OPTIMIZER

        if opt.lower() == 'adam':
            return self.adam_optimizer(lr_policy, loss)
        elif opt.lower() == 'sgd':
            return self.sgd_optimizer(lr_policy, loss)
        else:
            raise Exception(
                "unsupport optimizer solver, only support adam and sgd")

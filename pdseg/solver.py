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
import paddle.fluid as fluid
import numpy as np
import importlib
from utils.config import cfg
try:
    from paddle.fluid.contrib.mixed_precision.decorator import OptimizerWithMixedPrecison, decorate, AutoMixedPrecisionLists
except:
    from paddle.fluid.contrib.mixed_precision.decorator import OptimizerWithMixedPrecision, decorate, AutoMixedPrecisionLists


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

    def lr_warmup(self, learning_rate, start_lr, end_lr):
        linear_step = end_lr - start_lr
        lr = fluid.layers.tensor.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=True,
            name="learning_rate_warmup")

        global_step = fluid.layers.learning_rate_scheduler._decay_step_counter()
        warmup_counter = fluid.layers.autoincreased_step_counter(
            counter_name='@LR_DECAY_COUNTER_WARMUP_IN_SEG@', begin=1, step=1)
        global_counter = fluid.default_main_program().global_block(
        ).vars['@LR_DECAY_COUNTER@']
        warmup_counter = fluid.layers.cast(warmup_counter, 'float32')

        with fluid.layers.control_flow.Switch() as switch:
            with switch.case(warmup_counter <= self.warmup_step):
                decayed_lr = start_lr + linear_step * (
                    warmup_counter / self.warmup_step)
                fluid.layers.tensor.assign(decayed_lr, lr)
                # hold the global_step to 0 during the warm-up phase
                fluid.layers.increment(global_counter, value=-1)
            with switch.default():
                fluid.layers.tensor.assign(learning_rate, lr)
        return lr

    def piecewise_decay(self):
        gamma = cfg.SOLVER.GAMMA
        bd = [self.step_per_epoch * e for e in cfg.SOLVER.DECAY_EPOCH]
        lr = [cfg.SOLVER.LR * (gamma**i) for i in range(len(bd) + 1)]
        decayed_lr = fluid.layers.piecewise_decay(boundaries=bd, values=lr)
        return decayed_lr

    def poly_decay(self):
        power = cfg.SOLVER.POWER
        decayed_lr = fluid.layers.polynomial_decay(
            cfg.SOLVER.LR, self.decay_step, end_learning_rate=0, power=power)
        return decayed_lr

    def cosine_decay(self):
        decayed_lr = fluid.layers.cosine_decay(
            cfg.SOLVER.LR, self.step_per_epoch, self.decay_epochs)
        return decayed_lr

    def get_lr(self, lr_policy):
        if lr_policy.lower() == 'poly':
            decayed_lr = self.poly_decay()
        elif lr_policy.lower() == 'piecewise':
            decayed_lr = self.piecewise_decay()
        elif lr_policy.lower() == 'cosine':
            decayed_lr = self.cosine_decay()
        else:
            raise Exception(
                "unsupport learning decay policy! only support poly,piecewise,cosine"
            )

        decayed_lr = self.lr_warmup(decayed_lr, 0, cfg.SOLVER.LR)
        return decayed_lr

    def sgd_optimizer(self, lr_policy, loss):
        decayed_lr = self.get_lr(lr_policy)
        optimizer = fluid.optimizer.Momentum(
            learning_rate=decayed_lr,
            momentum=self.momentum,
            regularization=fluid.regularizer.L2Decay(
                regularization_coeff=self.weight_decay),
        )
        if cfg.MODEL.FP16:
            if cfg.MODEL.MODEL_NAME in ["pspnet"]:
                custom_black_list = {"pool2d"}
            else:
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

        optimizer.minimize(loss)
        return decayed_lr

    def adam_optimizer(self, lr_policy, loss):
        decayed_lr = self.get_lr(lr_policy)
        optimizer = fluid.optimizer.Adam(
            learning_rate=decayed_lr,
            beta1=self.momentum,
            beta2=self.momentum2,
            regularization=fluid.regularizer.L2Decay(
                regularization_coeff=self.weight_decay),
        )
        optimizer.minimize(loss)
        return decayed_lr

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

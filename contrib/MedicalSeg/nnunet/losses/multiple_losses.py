# Implementation of this model is borrowed and modified
# (from torch to paddle) from here:
# https://github.com/MIC-DKFZ/nnUNet

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

import pickle
import numpy as np

from paddle import nn

from medicalseg.cvlibs import manager


@manager.LOSSES.add_component
class MultipleLoss(nn.Layer):
    def __init__(self,
                 losses,
                 coef,
                 ignore_index=255,
                 plan_path=None,
                 stage=None):
        super(MultipleLoss, self).__init__()
        if not isinstance(losses, list):
            raise TypeError('`losses` must be a list!')
        if not isinstance(coef, list):
            raise TypeError('`coef` must be a list!')
        len_losses = len(losses)
        len_coef = len(coef)
        if len_losses != len_coef:
            raise ValueError(
                'The length of `losses` should equal to `coef`, but they are {} and {}.'
                .format(len_losses, len_coef))

        self.loss = losses[0]
        self.coef = coef
        self.stage = stage
        self.load_plans(plan_path)

        net_numpool = len(self.net_num_pool_op_kernel_sizes)

        weights = np.array([1 / (2**i) for i in range(net_numpool)])

        mask = np.array([True] + [
            True if i < net_numpool - 1 else False
            for i in range(1, net_numpool)
        ])
        weights[~mask] = 0
        weights = weights / weights.sum()
        self.weight_factors = weights

    def load_plans(self, plan_path):
        with open(plan_path, 'rb') as f:
            plans = pickle.load(f)
        if self.stage is None:
            assert len(list(plans['plans_per_stage'].keys())) == 1, \
                "If self.stage is None then there can be only one stage in the plans file. That seems to not be the " \
                "case. Please specify which stage of the cascade must be trained"
            self.stage = list(plans['plans_per_stage'].keys())[0]
        self.plans = plans

        stage_plans = self.plans['plans_per_stage'][self.stage]

        if 'pool_op_kernel_sizes' not in stage_plans.keys():
            assert 'num_pool_per_axis' in stage_plans.keys()
            self.print_to_log_file(
                "WARNING! old plans file with missing pool_op_kernel_sizes. Attempting to fix it..."
            )
            self.net_num_pool_op_kernel_sizes = []
            for i in range(max(self.net_pool_per_axis)):
                curr = []
                for j in self.net_pool_per_axis:
                    if (max(self.net_pool_per_axis) - j) <= i:
                        curr.append(2)
                    else:
                        curr.append(1)
                self.net_num_pool_op_kernel_sizes.append(curr)
        else:
            self.net_num_pool_op_kernel_sizes = stage_plans[
                'pool_op_kernel_sizes']

    def forward(self, logits, labels):
        assert isinstance(logits,
                          (tuple, list)), "x must be either tuple or list"
        assert isinstance(labels,
                          (tuple, list)), "y must be either tuple or list"
        per_channel_dice = None
        if self.weight_factors is None:
            weights = [1] * len(logits)
        else:
            weights = self.weight_factors

        l, per_channel_dice = self.loss(logits[0], labels[0])
        l = weights[0] * l
        for i in range(1, len(logits)):
            if weights[i] != 0:
                loss, _ = self.loss(logits[i], labels[i])
                l += weights[i] * loss
        return l, per_channel_dice

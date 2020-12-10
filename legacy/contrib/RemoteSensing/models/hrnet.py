#   Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
import paddle.fluid as fluid
import os
from os import path as osp
import numpy as np
from collections import OrderedDict
import copy
import math
import time
import tqdm
import cv2
import yaml
import utils
import utils.logging as logging
from utils.utils import seconds_to_hms, get_environ_info
from utils.metrics import ConfusionMatrix
import nets
import transforms.transforms as T
from .base import BaseModel


def dict2str(dict_input):
    out = ''
    for k, v in dict_input.items():
        try:
            v = round(float(v), 6)
        except:
            pass
        out = out + '{}={}, '.format(k, v)
    return out.strip(', ')


class HRNet(BaseModel):
    def __init__(self,
                 num_classes=2,
                 input_channel=3,
                 stage1_num_modules=1,
                 stage1_num_blocks=[4],
                 stage1_num_channels=[64],
                 stage2_num_modules=1,
                 stage2_num_blocks=[4, 4],
                 stage2_num_channels=[18, 36],
                 stage3_num_modules=4,
                 stage3_num_blocks=[4, 4, 4],
                 stage3_num_channels=[18, 36, 72],
                 stage4_num_modules=3,
                 stage4_num_blocks=[4, 4, 4, 4],
                 stage4_num_channels=[18, 36, 72, 144],
                 use_bce_loss=False,
                 use_dice_loss=False,
                 class_weight=None,
                 ignore_index=255,
                 sync_bn=True):
        super().__init__(
            num_classes=num_classes,
            use_bce_loss=use_bce_loss,
            use_dice_loss=use_dice_loss,
            class_weight=class_weight,
            ignore_index=ignore_index,
            sync_bn=sync_bn)
        self.init_params = locals()
        self.input_channel = input_channel
        self.stage1_num_modules = stage1_num_modules
        self.stage1_num_blocks = stage1_num_blocks
        self.stage1_num_channels = stage1_num_channels
        self.stage2_num_modules = stage2_num_modules
        self.stage2_num_blocks = stage2_num_blocks
        self.stage2_num_channels = stage2_num_channels
        self.stage3_num_modules = stage3_num_modules
        self.stage3_num_blocks = stage3_num_blocks
        self.stage3_num_channels = stage3_num_channels
        self.stage4_num_modules = stage4_num_modules
        self.stage4_num_blocks = stage4_num_blocks
        self.stage4_num_channels = stage4_num_channels

    def build_net(self, mode='train'):
        """应根据不同的情况进行构建"""
        model = nets.HRNet(
            self.num_classes,
            self.input_channel,
            mode=mode,
            stage1_num_modules=self.stage1_num_modules,
            stage1_num_blocks=self.stage1_num_blocks,
            stage1_num_channels=self.stage1_num_channels,
            stage2_num_modules=self.stage2_num_modules,
            stage2_num_blocks=self.stage2_num_blocks,
            stage2_num_channels=self.stage2_num_channels,
            stage3_num_modules=self.stage3_num_modules,
            stage3_num_blocks=self.stage3_num_blocks,
            stage3_num_channels=self.stage3_num_channels,
            stage4_num_modules=self.stage4_num_modules,
            stage4_num_blocks=self.stage4_num_blocks,
            stage4_num_channels=self.stage4_num_channels,
            use_bce_loss=self.use_bce_loss,
            use_dice_loss=self.use_dice_loss,
            class_weight=self.class_weight,
            ignore_index=self.ignore_index)
        inputs = model.generate_inputs()
        model_out = model.build_net(inputs)
        outputs = OrderedDict()
        if mode == 'train':
            self.optimizer.minimize(model_out)
            outputs['loss'] = model_out
        else:
            outputs['pred'] = model_out[0]
            outputs['logit'] = model_out[1]
        return inputs, outputs

    def train(self,
              num_epochs,
              train_reader,
              train_batch_size=2,
              eval_reader=None,
              eval_best_metric='miou',
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights=None,
              resume_weights=None,
              optimizer=None,
              learning_rate=0.01,
              lr_decay_power=0.9,
              regularization_coeff=5e-4,
              use_vdl=False):
        super().train(
            num_epochs=num_epochs,
            train_reader=train_reader,
            train_batch_size=train_batch_size,
            eval_reader=eval_reader,
            eval_best_metric=eval_best_metric,
            save_interval_epochs=save_interval_epochs,
            log_interval_steps=log_interval_steps,
            save_dir=save_dir,
            pretrain_weights=pretrain_weights,
            resume_weights=resume_weights,
            optimizer=optimizer,
            learning_rate=learning_rate,
            lr_decay_power=lr_decay_power,
            regularization_coeff=regularization_coeff,
            use_vdl=use_vdl)

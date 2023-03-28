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

# In this package we provide PaddleSeg-style functions for training, 
# validation, prediction, and inference.
# We do not re-use the APIs of PaddleSeg because there is a large gap
# between semantic segmentation and panoptic segmentation.

import abc

import paddle

__all__ = ['AMPLauncher']


class _Launcher(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train_step(self):
        pass

    @abc.abstractmethod
    def infer_step(self):
        pass

    @classmethod
    def __subclasshook__(cls, subclass):
        if cls is _Launcher:
            return (hasattr(subclass, 'train_step') and
                    callable(subclass.train_step) and
                    hasattr(subclass, 'infer_step') and
                    callable(subclass.infer_step))
        return NotImplemented


class AMPLauncher(_Launcher):
    def __init__(self, runner, precision='fp32', amp_level='O1'):
        super().__init__()
        self.runner = runner
        if precision not in ('fp32', 'fp16'):
            raise ValueError("`precision` must be 'fp32' or 'fp16'.")
        self.precision = precision
        if amp_level not in ('O1', 'O2'):
            raise ValueError("`amp_level` must be 'O1' or 'O2'.")
        self.amp_level = amp_level
        if self.precision == 'fp16' and self.runner.optimizer is not None:
            self.scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        if (self.amp_level == 'O2' and self.runner.model is not None and
                self.runner.optimizer is not None):
            self.model, self.optimizer = paddle.amp.decorate(
                models=self.runner.model,
                optimizers=self.runner.optimizer,
                level='O2',
                save_dtype='float32')
        else:
            self.model = self.runner.model
            self.optimizer = self.runner.optimizer

    def train_step(self, data, return_loss_list=False):
        if self.precision == 'fp32':
            loss, loss_list = self.train_step_fp32(data)
        elif self.precision == 'fp16':
            loss, loss_list = self.train_step_fp16(data)
        self.runner.update_lr(loss)
        self.model.clear_gradients()
        if return_loss_list:
            return loss, loss_list
        else:
            return loss

    def train_step_fp32(self, data):
        # Forward
        net_out = self.runner.train_forward(data)

        # Compute loss
        loss_list = self.runner.compute_losses(net_out, data)
        loss = self.runner.aggregate_losses(loss_list)

        # Backward
        loss.backward()

        # Update parameters
        self.optimizer.step()
        return loss, loss_list

    def train_step_fp16(self, data):
        with paddle.amp.auto_cast(
                level=self.amp_level,
                enable=True,
                custom_white_list={
                    'elementwise_add', 'batch_norm', 'sync_batch_norm'
                },
                custom_black_list={'bilinear_interp_v2'}):
            # Forward
            net_out = self.runner.train_forward(data)
            # Compute loss
            loss_list = self.runner.compute_losses(net_out, data)
            loss = self.runner.aggregate_losses(loss_list)

        # Backward
        scaled = self.scaler.scale(loss)
        scaled.backward()

        # Update parameters
        self.scaler.minimize(self.optimizer, scaled)
        return loss, loss_list

    def infer_step(self, data):
        if self.precision == 'fp32':
            pp_out = self.infer_step_fp32(data)
        elif self.precision == 'fp16':
            pp_out = self.infer_step_fp16(data)
        return pp_out

    def infer_step_fp32(self, data):
        return self.runner.infer_forward(data)

    def infer_step_fp16(self, data):
        with paddle.amp.auto_cast(
                level=self.amp_level,
                enable=True,
                custom_white_list={
                    'elementwise_add', 'batch_norm', 'sync_batch_norm'
                },
                custom_black_list={'bilinear_interp_v2'}):
            pp_out = self.runner.infer_forward(data)
        return pp_out

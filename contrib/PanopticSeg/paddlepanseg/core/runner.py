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

from paddlepanseg.core import infer

__all__ = ['PanSegRunner', 'EMPTY']


class _Empty(object):
    pass


EMPTY = _Empty()


class Runner(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update_lr(self, loss):
        pass

    @abc.abstractmethod
    def train_forward(self, data):
        pass

    @abc.abstractmethod
    def infer_forward(self, data):
        pass

    @abc.abstractmethod
    def compute_losses(self, net_out, data):
        pass

    @abc.abstractmethod
    def aggregate_losses(self, loss_list):
        pass


class _LazyBindingMixin(object):
    ATTRS = ()

    def __init__(self):
        super().__init__()
        for attr in self.ATTRS:
            setattr(self, attr, EMPTY)

    def bind(self, **attrs):
        for key, val in attrs.items():
            if key not in self.ATTRS:
                raise AttributeError
            else:
                setattr(self, key, val)
        self._post_bind()

    def _post_bind(self):
        # Do nothing
        pass


class PanSegRunner(Runner, _LazyBindingMixin):
    ATTRS = ('model', 'criteria', 'optimizer', 'postprocessor')

    def update_lr(self, loss):
        lr_sche = self.optimizer._learning_rate
        if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
            if isinstance(lr_sche, paddle.optimizer.lr.ReduceOnPlateau):
                lr_sche.step(loss)
            else:
                lr_sche.step()

    def train_forward(self, data):
        images = data['img']
        net_out = self.model(images)
        return net_out

    def infer_forward(self, data):
        pp_out = infer.inference(
            model=self.model, data=data, postprocessor=self.postprocessor)
        return pp_out

    def compute_losses(self, net_out, data):
        loss_list = []
        for i in range(len(self.criteria['types'])):
            loss_i = self.criteria['types'][i]
            coef_i = self.criteria['coef'][i]
            # Do a 'literal' check to find `MixedLoss` objects
            if loss_i.__class__.__name__ == 'MixedLoss':
                mixed_loss_list = loss_i(data, net_out)
                for mixed_loss in mixed_loss_list:
                    loss_list.append(coef_i * mixed_loss)
            else:
                loss_list.append(coef_i * loss_i(data, net_out))
        return loss_list

    def aggregate_losses(self, loss_list):
        return sum(loss_list)

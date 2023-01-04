# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

from paddleseg.cvlibs import manager
from paddleseg.utils import logger


class BaseOptimizer(object):
    """
    Base optimizer in PaddleSeg.

    Args:
        weight_decay(float, optional): A float value as coeff of L2 regularization.
        custom_params (dict, optional): custom_params specify different options for
            different parameter groups such as the learning rate and weight decay.
    """

    def __init__(self, weight_decay=None, custom_params=None):
        if weight_decay is not None:
            assert isinstance(weight_decay, float), \
                "`weight_decay` must be a float."
        if custom_params is not None:
            assert isinstance(custom_params, list), \
                "`custom_params` must be a list."
            for item in custom_params:
                assert isinstance(item, dict), \
                "The item of `custom_params` must be a dict"
        self.weight_decay = weight_decay
        self.custom_params = custom_params
        self.args = {'weight_decay': weight_decay}

    def __call__(self, model, lr):
        # Create optimizer
        pass

    def _model_params(self, model):
        # Collect different parameter groups
        if self.custom_params is None or len(self.custom_params) == 0:
            return model.parameters()

        groups_num = len(self.custom_params) + 1
        params_list = [[] for _ in range(groups_num)]
        for name, param in model.named_parameters():
            if param.stop_gradient:
                continue
            for idx, item in enumerate(self.custom_params):
                if item['name'] in name:
                    params_list[idx].append(param)
                    break
            else:
                params_list[-1].append(param)

        res = []
        for idx, item in enumerate(self.custom_params):
            lr_mult = item.get("lr_mult", 1.0)
            weight_decay_mult = item.get("weight_decay_mult", None)
            param_dict = {'params': params_list[idx], 'learning_rate': lr_mult}
            if self.weight_decay is not None and weight_decay_mult is not None:
                param_dict[
                    'weight_decay'] = self.weight_decay * weight_decay_mult
            res.append(param_dict)
        res.append({'params': params_list[-1]})

        msg = 'Parameter groups for optimizer: \n'
        for idx, item in enumerate(self.custom_params):
            params_name = [p.name for p in params_list[idx]]
            item = item.copy()
            item['params_name'] = params_name
            msg += 'Group {}: \n{} \n'.format(idx, item)
        msg += 'Last group:\n params_name: {}'.format(
            [p.name for p in params_list[-1]])
        logger.info(msg)

        return res


@manager.OPTIMIZER.add_component
class SGD(BaseOptimizer):
    """
    SGD optimizer. 

    An example in config:
    `
    optimizer:
      type: SGD
      weight_decay: 4.0e-5
      custom_params:
        - name: backbone
          lr_mult: 0.1
    `
    """

    def __init__(self, weight_decay=None, custom_params=None):
        super().__init__(weight_decay=weight_decay, custom_params=custom_params)

    def __call__(self, model, lr):
        params = self._model_params(model)
        return paddle.optimizer.SGD(learning_rate=lr,
                                    parameters=params,
                                    **self.args)


@manager.OPTIMIZER.add_component
class Momentum(BaseOptimizer):
    """
    Momentum optimizer. 
    """

    def __init__(self,
                 momentum=0.9,
                 use_nesterov=False,
                 weight_decay=None,
                 custom_params=None):
        super().__init__(weight_decay=weight_decay, custom_params=custom_params)
        self.args.update({'momentum': momentum, 'use_nesterov': use_nesterov})

    def __call__(self, model, lr):
        params = self._model_params(model)
        return paddle.optimizer.Momentum(
            learning_rate=lr, parameters=params, **self.args)


@manager.OPTIMIZER.add_component
class Adam(BaseOptimizer):
    """
    Adam optimizer. 
    """

    def __init__(self,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-08,
                 lazy_mode=False,
                 weight_decay=None,
                 custom_params=None):
        super().__init__(weight_decay=weight_decay, custom_params=custom_params)
        self.args.update({
            'beta1': beta1,
            'beta2': beta2,
            'epsilon': epsilon,
            'lazy_mode': lazy_mode
        })

    def __call__(self, model, lr):
        params = self._model_params(model)
        opt = paddle.optimizer.Adam(
            learning_rate=lr, parameters=params, **self.args)
        return opt


@manager.OPTIMIZER.add_component
class AdamW(BaseOptimizer):
    """
    AdamW optimizer. 
    """

    def __init__(self,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-08,
                 weight_decay=0.01,
                 lazy_mode=False,
                 custom_params=None):
        super().__init__(weight_decay=weight_decay, custom_params=custom_params)
        self.args.update({
            'beta1': beta1,
            'beta2': beta2,
            'epsilon': epsilon,
            'lazy_mode': lazy_mode
        })

    def __call__(self, model, lr):
        params = self._model_params(model)
        opt = paddle.optimizer.AdamW(
            learning_rate=lr, parameters=params, **self.args)
        return opt

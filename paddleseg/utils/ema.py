# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle

class EMA(object):
    """
    The implementation of Exponential Moving Average for the trainable parameters.

    Args:
        model (nn.Layer): The model for applying EMA.
        decay (float, optional): Decay is used to calculate ema_variable by
            `ema_variable = decay * ema_variable + (1 - decay) * new_variable`.
            Default: 0.99.
    
    Returns:
        None
    
    Examples:
        .. code-block:: python

            # 1. Define model and dataset
        
            # 2. Create EMA
            ema = EMA(model, decay=0.99)

            # 3. Train stage
            for data in dataloader():
                ...
                optimizer.step()
                ema.step()

            # 4. Evaluate stage
            ema.apply()     # Use the EMA data to replace the origin data

            for data in dataloader():
                ...
            
            ema.restore()   # Restore the origin data to the model

    """
    def __init__(self, model, decay=0.99):
        super().__init__()

        assert isinstance(model, paddle.nn.Layer), \
            "The model should be the instance of paddle.nn.Layer."
        assert decay >= 0 and decay <= 1.0, \
            "The decay = {} should in [0.0, 1.0]".format(decay)

        self._model = model
        self._decay = decay
        self._ema_data = {}
        self._backup_data = {}

        for name, param in self._model.named_parameters():
            if not param.stop_gradient:
                self._ema_data[name] = param.numpy()

    def step(self):
        """
        Calculate the EMA data for all trainable parameters.
        """
        for name, param in self._model.named_parameters():
            if not param.stop_gradient:
                assert name in self._ema_data, \
                    "The param ({}) isn't in the model".format(name)
                self._ema_data[name] = self._decay * self._ema_data[name] \
                    + (1.0 - self._decay) * param.numpy()

    def apply(self):
        """
        Save the origin data and use the EMA data to replace the origin data.
        """
        for name, param in self._model.named_parameters():
            if not param.stop_gradient:
                assert name in self._ema_data, \
                    "The param ({}) isn't in the model".format(name)
                self._backup_data[name] = param.numpy()
                param.set_value(self._ema_data[name])

    def restore(self):
        """
        Restore the origin data to the model.
        """
        for name, param in self._model.named_parameters():
            if not param.stop_gradient:
                assert name in self._backup_data, \
                    "The param ({}) isn't in the model".format(name)
                param.set_value(self._backup_data[name])
        self._backup_data = {}

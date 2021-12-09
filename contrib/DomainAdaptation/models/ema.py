# This file is made available under Apache License, Version 2.0
# This file is based on code available under the MIT License here:
# https://github.com/valencebond/FixMatch_pytorch/blob/master/models/ema.py
#
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


class EMA(object):
    def __init__(self, model, alpha=0.999):
        """ Model exponential moving average. """
        self.step = 0
        self.model = model
        self.alpha = alpha
        self.shadow = self.get_model_state()  # init model
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self):
        decay = self.alpha
        state = self.model.state_dict()  # current params
        for name in self.param_keys:
            self.shadow[name] = decay * self.shadow[name] + (
                1 - decay) * state[name].detach().clone()
        self.step += 1

    def update_buffer(self):
        # No EMA for buffer values (for now)
        state = self.model.state_dict()
        for name in self.buffer_keys:
            self.shadow[name] = state[name].detach().clone()

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.set_dict(self.shadow)

    def restore(self):
        self.model.load_dict(self.backup)

    def get_model_state(self):
        """get model current statre """
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }

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

import os

import paddle

from paddleseg.utils import logger

__all__ = ['cps_resume', 'get_in_channels', 'set_in_channels']


def cps_resume(model, optimizer_l, optimizer_r, resume_model):
    if resume_model is not None:
        logger.info('Resume model from {}'.format(resume_model))
        if os.path.exists(resume_model):
            resume_model = os.path.normpath(resume_model)
            model_path = os.path.join(resume_model, 'model.pdparams')
            para_state_dict = paddle.load(model_path)

            opti_l_path = os.path.join(resume_model, 'model_l.pdopt')
            opti_l_state_dict = paddle.load(opti_l_path)

            opti_r_path = os.path.join(resume_model, 'model_r.pdopt')
            opti_r_state_dict = paddle.load(opti_r_path)

            model.set_state_dict(para_state_dict)
            optimizer_l.set_state_dict(opti_l_state_dict)
            optimizer_r.set_state_dict(opti_l_state_dict)

            epoch = resume_model.split('_')[-1]
            epoch = int(epoch)
            return epoch
        else:
            raise ValueError(
                'Directory of the model needed to resume is not Found: {}'.
                format(resume_model))
    else:
        logger.info('No model needed to resume.')


def get_in_channels(model_cfg):
    if 'backbone_l' in model_cfg:
        return model_cfg['backbone_l'].get('in_channels', None)
    else:
        return model_cfg.get('in_channels', None)


def set_in_channels(model_cfg, in_channels):
    model_cfg = model_cfg.copy()
    if ('backbone_l' in model_cfg) and ('backbone_r' in model_cfg):
        model_cfg['backbone_l']['in_channels'] = in_channels
        model_cfg['backbone_r']['in_channels'] = in_channels
    else:
        model_cfg['in_channels'] = in_channels

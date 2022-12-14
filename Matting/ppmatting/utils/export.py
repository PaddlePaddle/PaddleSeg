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


def get_input_spec(model_name, shape, trimap):
    """
    Get the input spec accoring the model_name.

    Args:
        model_name (str): The model name
        shape (str): The shape of input image
        trimap (str): Whether a trimap is required

    """
    input_spec = [{"img": paddle.static.InputSpec(shape=shape, name='img')}]
    if trimap:
        shape[1] = 1
        input_spec[0]['trimap'] = paddle.static.InputSpec(
            shape=shape, name='trimap')

    if model_name == 'RVM':
        input_spec.append(
            paddle.static.InputSpec(
                shape=[None, 16, None, None], name='r1'))
        input_spec.append(
            paddle.static.InputSpec(
                shape=[None, 20, None, None], name='r2'))
        input_spec.append(
            paddle.static.InputSpec(
                shape=[None, 40, None, None], name='r3'))
        input_spec.append(
            paddle.static.InputSpec(
                shape=[None, 64, None, None], name='r4'))
        input_spec.append(
            paddle.static.InputSpec(
                shape=[1], name='downsample_ratio'))
    return input_spec

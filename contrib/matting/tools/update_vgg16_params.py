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

import paddle


def update_vgg16_params(model_path):
    param_state_dict = paddle.load(model_path)
    # first conv weight name _conv_block_1._conv_1.weight, shape is [64, 3, ,3, 3]
    # first fc weight name: _fc1.weight, shape is [25088, 4096]
    for k, v in param_state_dict.items():
        print(k, v.shape)

    # # first weight
    weight = param_state_dict['_conv_block_1._conv_1.weight']  # [64, 3,3,3]
    print('ori shape: ', weight.shape)
    zeros_pad = paddle.zeros((64, 1, 3, 3))
    param_state_dict['_conv_block_1._conv_1.weight'] = paddle.concat(
        [weight, zeros_pad], axis=1)
    print('shape after padding',
          param_state_dict['_conv_block_1._conv_1.weight'].shape)

    # fc1
    weight = param_state_dict['_fc1.weight']
    weight = paddle.transpose(weight, [1, 0])
    print('after transpose: ', weight.shape)
    weight = paddle.reshape(weight, (4096, 512, 7, 7))
    print('after reshape: ', weight.shape)
    weight = weight[0:512, :, 2:5, 2:5]
    print('after crop: ', weight.shape)
    param_state_dict['_conv_6.weight'] = weight

    del param_state_dict['_fc1.weight']
    del param_state_dict['_fc1.bias']
    del param_state_dict['_fc2.weight']
    del param_state_dict['_fc2.bias']
    del param_state_dict['_out.weight']
    del param_state_dict['_out.bias']

    paddle.save(param_state_dict, 'VGG16_pretrained.pdparams')


if __name__ == "__main__":
    paddle.set_device('cpu')
    model_path = '~/.paddleseg/pretrained_model/dygraph/VGG16_pretrained.pdparams'
    update_vgg16_params(model_path)

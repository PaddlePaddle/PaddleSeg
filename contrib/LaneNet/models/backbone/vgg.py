#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.fluid as fluid
from paddle.fluid import ParamAttr

__all__ = ["VGGNet", "VGG11", "VGG13", "VGG16", "VGG19"]


class VGGNet():
    def __init__(self, layers=16):
        self.layers = layers

    def net(self, input, class_dim=1000):
        layers = self.layers
        vgg_spec = {
            11: ([1, 1, 2, 2, 2]),
            13: ([2, 2, 2, 2, 2]),
            16: ([2, 2, 3, 3, 3]),
            19: ([2, 2, 4, 4, 4])
        }
        assert layers in vgg_spec.keys(), \
            "supported layers are {} but input layer is {}".format(vgg_spec.keys(), layers)

        fetures_dict = {}
        nums = vgg_spec[layers]
        conv1 = self.conv_block(input, 64, nums[0], name="conv1_")
        conv2 = self.conv_block(conv1, 128, nums[1], name="conv2_")
        conv3 = self.conv_block(conv2, 256, nums[2], name="conv3_")
        conv4 = self.conv_block(conv3, 512, nums[3], name="conv4_")
        conv5 = self.conv_block(conv4, 512, nums[4], name="conv5_")
        fetures_dict['pool3'] = conv3
        fetures_dict['pool4'] = conv4
        fetures_dict['pool5'] = conv5
        return fetures_dict
        fc_dim = 4096
        fc_name = ["fc6", "fc7", "fc8"]
        fc1 = fluid.layers.fc(
            input=conv5,
            size=fc_dim,
            act='relu',
            param_attr=fluid.param_attr.ParamAttr(name=fc_name[0] + "_weights"),
            bias_attr=fluid.param_attr.ParamAttr(name=fc_name[0] + "_offset"))
        fc1 = fluid.layers.dropout(x=fc1, dropout_prob=0.5)
        fc2 = fluid.layers.fc(
            input=fc1,
            size=fc_dim,
            act='relu',
            param_attr=fluid.param_attr.ParamAttr(name=fc_name[1] + "_weights"),
            bias_attr=fluid.param_attr.ParamAttr(name=fc_name[1] + "_offset"))
        fc2 = fluid.layers.dropout(x=fc2, dropout_prob=0.5)
        out = fluid.layers.fc(
            input=fc2,
            size=class_dim,
            param_attr=fluid.param_attr.ParamAttr(name=fc_name[2] + "_weights"),
            bias_attr=fluid.param_attr.ParamAttr(name=fc_name[2] + "_offset"))

        return out

    def conv_block(self, input, num_filter, groups, name=None):
        conv = input
        for i in range(groups):
            conv = fluid.layers.conv2d(
                input=conv,
                num_filters=num_filter,
                filter_size=3,
                stride=1,
                padding=1,
                act='relu',
                param_attr=fluid.param_attr.ParamAttr(
                    name=name + str(i + 1) + "_weights"),
                bias_attr=False)
        return fluid.layers.pool2d(
            input=conv, pool_size=2, pool_type='max', pool_stride=2)


def VGG11():
    model = VGGNet(layers=11)
    return model


def VGG13():
    model = VGGNet(layers=13)
    return model


def VGG16():
    model = VGGNet(layers=16)
    return model


def VGG19():
    model = VGGNet(layers=19)
    return model

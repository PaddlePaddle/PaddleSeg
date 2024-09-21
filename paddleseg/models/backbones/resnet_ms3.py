# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
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
import paddle.nn as nn
from paddle.vision.models.resnet import resnet18, resnet34, resnet50, resnet101

from paddleseg.cvlibs import manager

__all__ = ["ResNet_MS3", "ResNet18", "ResNet34", "ResNet50", "ResNet101"]

model_dict = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
}


@manager.BACKBONES.add_component
class ResNet_MS3(nn.Layer):

    def __init__(self, pretrained=None, arch=''):
        super(ResNet_MS3, self).__init__()
        assert arch in model_dict.keys(), '{} not supported'.format(arch)
        self.model_name = arch
        self.pretrained = pretrained
        net = model_dict[arch](pretrained=pretrained)
        self.model = paddle.nn.Sequential(*(list(net.children())[:-2]))

    def forward(self, x):
        res = []
        for name, module in self.model._sub_layers.items():
            x = module(x)
            if name in ['4', '5', '6']:
                res.append(x)
        return res


@manager.BACKBONES.add_component
def ResNet18(**kwargs):
    return ResNet_MS3(pretrained=False, arch="resnet18")


@manager.BACKBONES.add_component
def ResNet34(**kwargs):
    return ResNet_MS3(pretrained=False, arch="resnet34")


@manager.BACKBONES.add_component
def ResNet50(**kwargs):
    return ResNet_MS3(pretrained=False, arch="resnet50")


@manager.BACKBONES.add_component
def ResNet101(**kwargs):
    return ResNet_MS3(pretrained=False, arch="resnet101")

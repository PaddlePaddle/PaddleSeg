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

import os

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils, logger

from .gscnn import GSCNNHead

__all__ = [
    'DeepLabV2',
]


@manager.MODELS.add_component
class DeepLabV2(nn.Layer):
    """
    The DeepLabV2 implementation based on PaddlePaddle.

    The original article refers to:

    Args:
        num_classes (int): The unique number of target classes.
        backbone (paddle.nn.Layer): Backbone network, currently support Resnet50_vd/Resnet101_vd/Xception65.
        backbone_indices (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
           Default: (0, 3).
        aspp_ratios (tuple, optional): The dilation rate using in ASSP module.
            If output_stride=16, aspp_ratios should be set as (1, 6, 12, 18).
            If output_stride=8, aspp_ratios is (1, 12, 24, 36).
            Default: (1, 6, 12, 18).
        aspp_out_channels (int, optional): The output channels of ASPP module. Default: 256.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
        data_format(str, optional): Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(0, 1, 2, 3),
                 aspp_ratios=(1, 6, 12, 18),
                 aspp_out_channels=256,
                 align_corners=False,
                 pretrained=None,
                 shape_stream=False):
        super().__init__()

        self.backbone = backbone
        self.shape_stream = shape_stream
        if shape_stream:
            print('####### ADD SHAPE STREAM ########')
        # backbone_channels = self.backbone.feat_channels
        # self.head = GSCNNHead(num_classes, backbone_indices, backbone_channels,
        #                       aspp_ratios, aspp_out_channels, align_corners)
        self.head = edge_branch(
            inplanes=(64, 256, 512, 1024, 2048),
            out_channels=2560,
            dilation_series=[6, 12, 18, 24],
            padding_series=[6, 12, 18, 24],
            num_classes=2)

        self.align_corners = align_corners  # should be true
        self.pretrained = pretrained  # should not load layer5
        self.init_weight()

    def forward(self, x):
        feat_list = self.backbone(x)
        # for item in feat_list:
        #     print('feat shape', item.shape) # align should be true, true, false

        if self.shape_stream:
            # logit_list = self.head(x, feat_list[2:], self.backbone.conv1_logit)
            logit_list = self.head(self.backbone.conv1_logit, *feat_list[2:])
            logit_list.extend(feat_list[:2])
            edge_logit, seg_logit, aug_logit = [
                F.interpolate(
                    logit,
                    x.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for logit in logit_list
            ]
            return [seg_logit, aug_logit, edge_logit]
        else:
            return [
                F.interpolate(
                    logit,
                    x.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for logit in feat_list[:2]
            ]

    def init_weight(self):
        if self.pretrained is not None:
            para_state_dict = paddle.load(self.pretrained)
            model_state_dict = self.backbone.state_dict()
            keys = model_state_dict.keys()
            num_params_loaded = 0
            for k in keys:
                k_parts = k.split('.')
                torchkey = 'backbone.' + k
                if k_parts[1] == 'layer5':
                    logger.warning("{} should not be loaded".format(k))
                elif torchkey not in para_state_dict:
                    logger.warning("{} is not in pretrained model".format(k))
                elif list(para_state_dict[torchkey].shape) != list(
                        model_state_dict[k].shape):
                    logger.warning(
                        "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                        .format(k, para_state_dict[torchkey].shape,
                                model_state_dict[k].shape))
                else:
                    model_state_dict[k] = para_state_dict[torchkey]
                    num_params_loaded += 1
            self.backbone.set_dict(model_state_dict)
            logger.info("There are {}/{} variables loaded into {}.".format(
                num_params_loaded, len(model_state_dict),
                self.backbone.__class__.__name__))


from paddleseg.models.backbones.resnet_vd import ConvBNLayer


class edge_branch(nn.Layer):
    def __init__(self, inplanes, out_channels, dilation_series, padding_series,
                 num_classes):
        super(edge_branch, self).__init__()
        self.conv_logit = nn.Conv2D(inplanes[0], 512, kernel_size=3)
        self.conv_x1 = nn.Conv2D(inplanes[1], 512, kernel_size=3)
        self.conv_x2 = nn.Conv2D(inplanes[2], 512, kernel_size=3)
        self.conv_x3 = nn.Conv2D(inplanes[3], 512, kernel_size=3)
        self.conv_x4 = nn.Conv2D(inplanes[4], 512, kernel_size=3)

        self.conv0 = ConvBNLayer(
            in_channels=512 * 5,
            out_channels=out_channels,
            kernel_size=3,
            act='relu')
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            act=None)

        self.add = layers.Add()
        self.relu = layers.Activation(act="relu")

        self.conv2d_list = nn.LayerList()
        for dilation, padding in zip(dilation_series, padding_series):
            weight_attr = paddle.ParamAttr(
                initializer=nn.initializer.Normal(std=0.01), learning_rate=10.0)
            bias_attr = paddle.ParamAttr(
                initializer=nn.initializer.Constant(value=0.0),
                learning_rate=10.0)
            self.conv2d_list.append(
                nn.Conv2D(
                    out_channels,
                    num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    weight_attr=weight_attr,
                    bias_attr=bias_attr))

    def forward(self, conv1_logit, x1, x2, x3, x4):
        H = paddle.shape(x4)[2]
        W = paddle.shape(x4)[3]
        conv1_logit = F.interpolate(
            conv1_logit, size=[H, W], mode='bilinear', align_corners=False)
        x1 = F.interpolate(x1, size=[H, W], mode='bilinear', align_corners=True)

        conv1_logit = self.conv_logit(conv1_logit)
        x1 = self.conv_x1(x1)
        x2 = self.conv_x2(x2)
        x3 = self.conv_x3(x3)
        x4 = self.conv_x4(x4)  # 1, 512, 81,161

        feats = paddle.concat([conv1_logit, x1, x2, x3, x4], axis=1)
        y = self.conv0(feats)
        y = self.conv1(y)

        y = self.add(feats, y)
        y = self.relu(y)

        out = self.conv2d_list[0](y)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](y)

        return [
            out,
        ]


if __name__ == '__main__':
    import numpy as np
    from backbones import ResNet101
    import reprod_log

    paddle.set_printoptions(precision=15)

    model = DeepLabV2(
        num_classes=19,
        backbone=ResNet101(num_classes=19),
        pretrained=
        '/ssd2/tangshiyu/Code/PaddleSeg/contrib/DomainAdaptation/models/torch_transfer_trained.pdparams',
        shape_stream=True)

    src_data, src_label = paddle.to_tensor(
        np.load('/ssd2/tangshiyu/Code/fake_data_src.npy')), np.load(
            '/ssd2/tangshiyu/Code/fake_label_src.npy')
    print(src_data.shape)
    # tgt_data, tgt_label = paddle.to_tensor(np.load('/ssd2/tangshiyu/Code/fake_data_tgt.npy')), np.load('/ssd2/tangshiyu/Code/fake_label_tgt.npy')

    out = model(src_data)
    for item in out:
        print(item.shape)  # 回归 640， 1280，
    # res_src = out[0].mean() + out[1].mean()
    # res_src.backward()
    # out = model(tgt_data)
    # res_tgt = out[0].mean()+out[1].mean()

    # reprod_logger = reprod_log.ReprodLogger()
    # reprod_logger.add("res_src", res_src.cpu().detach().numpy())
    # reprod_logger.add("res_tgt", res_tgt.cpu().detach().numpy())
    # reprod_logger.save("/ssd2/tangshiyu/Code/pixmatch/models/forward_paddle.npy")

    # print(out[0].mean())  # [-616.42187500]
    # print(out[1].mean())  # [4520.48437500]

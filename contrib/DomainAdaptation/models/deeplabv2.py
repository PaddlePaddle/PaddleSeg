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
            print('####### ADD GSCNN HEAD ########')
        backbone_channels = self.backbone.feat_channels
        self.head = GSCNNHead(num_classes, backbone_indices, backbone_channels,
                              aspp_ratios, aspp_out_channels, align_corners)

        self.align_corners = align_corners  # should be true
        self.pretrained = pretrained  # should not load layer5
        self.init_weight()

    def forward(self, x):
        feat_list = self.backbone(x)
        if self.shape_stream:
            logit_list = self.head(x, feat_list[:4], self.backbone.conv1_logit)
            logit_list.append(feat_list[-1])
            seg_logit, edge_logit, aug_logit = [
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
                    align_corners=self.align_corners) for logit in feat_list
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


if __name__ == '__main__':
    import numpy as np
    from backbones import ResNet101
    import reprod_log

    paddle.set_printoptions(precision=15)

    model = DeepLabV2(
        num_classes=19,
        backbone=ResNet101(num_classes=19),
        pretrained=
        '/ssd2/tangshiyu/Code/PaddleSeg/contrib/DomainAdaptation/models/torch_transfer_trained.pdparams'
    )

    src_data, src_label = paddle.to_tensor(
        np.load('/ssd2/tangshiyu/Code/fake_data_src.npy')), np.load(
            '/ssd2/tangshiyu/Code/fake_label_src.npy')
    # tgt_data, tgt_label = paddle.to_tensor(np.load('/ssd2/tangshiyu/Code/fake_data_tgt.npy')), np.load('/ssd2/tangshiyu/Code/fake_label_tgt.npy')

    out = model(src_data)
    res_src = out[0].mean() + out[1].mean()
    res_src.backward()
    # out = model(tgt_data)
    # res_tgt = out[0].mean()+out[1].mean()

    # reprod_logger = reprod_log.ReprodLogger()
    # reprod_logger.add("res_src", res_src.cpu().detach().numpy())
    # reprod_logger.add("res_tgt", res_tgt.cpu().detach().numpy())
    # reprod_logger.save("/ssd2/tangshiyu/Code/pixmatch/models/forward_paddle.npy")

    # print(out[0].mean())  # [-616.42187500]
    # print(out[1].mean())  # [4520.48437500]

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.nn.functional as F
from paddle import fluid
from paddle.fluid.dygraph import Conv2D

from dygraph.cvlibs import manager
from dygraph.models import model_utils
from dygraph.models.architectures import layer_utils
from dygraph.utils import utils


class PSPNet(fluid.dygraph.Layer):
    """
    The PSPNet implementation

    The orginal artile refers to 
        Zhao, Hengshuang, et al. "Pyramid scene parsing network." 
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
        (https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf)

    Args:
        backbone (str): backbone name, currently support Resnet50/101.

        num_classes (int): the unique number of target classes. Default 2.

        output_stride (int): the ratio of input size and final feature size. Default 16.

        backbone_indices (tuple): two values in the tuple indicte the indices of output of backbone.
                        the first index will be taken as a deep-supervision feature in auxiliary layer;
                        the second one will be taken as input of Pyramid Pooling Module (PPModule).
                        Usually backbone consists of four downsampling stage, and return an output of
                        each stage, so we set default (2, 3), which means taking feature map of the third
                        stage (res4b22) in backbone, and feature map of the fourth stage (res5c) as input of PPModule.

        backbone_channels (tuple): the same length with "backbone_indices". It indicates the channels of corresponding index.

        pp_out_channels (int): output channels after Pyramid Pooling Module. Default to 1024.

        bin_sizes (tuple): the out size of pooled feature maps. Default to (1,2,3,6).

        enable_auxiliary_loss (bool): a bool values indictes whether adding auxiliary loss. Default to True.

        ignore_index (int): the value of ground-truth mask would be ignored while doing evaluation. Default to 255.

        pretrained_model (str): the pretrained_model path of backbone.
    """

    def __init__(self,
                 backbone,
                 num_classes=2,
                 output_stride=16,
                 backbone_indices=(2, 3),
                 backbone_channels=(1024, 2048),
                 pp_out_channels=1024,
                 bin_sizes=(1, 2, 3, 6),
                 enable_auxiliary_loss=True,
                 ignore_index=255,
                 pretrained_model=None):

        super(PSPNet, self).__init__()
        self.backbone = manager.BACKBONES[backbone](output_stride=output_stride,
                                                    multi_grid=(1, 1, 1))
        self.backbone_indices = backbone_indices

        self.psp_module = PPModule(in_channels=backbone_channels[1],
                                   out_channels=pp_out_channels,
                                   bin_sizes=bin_sizes)

        self.conv = Conv2D(num_channels=pp_out_channels,
                           num_filters=num_classes,
                           filter_size=1)

        if enable_auxiliary_loss:
            self.fcn_head = model_utils.FCNHead(in_channels=backbone_channels[0], out_channels=num_classes)

        self.enable_auxiliary_loss = enable_auxiliary_loss
        self.ignore_index = ignore_index

        self.init_weight(pretrained_model)

    def forward(self, input, label=None):

        _, feat_list = self.backbone(input)

        x = feat_list[self.backbone_indices[1]]
        x = self.psp_module(x)
        x = F.dropout(x, dropout_prob=0.1)
        logit = self.conv(x)
        logit = fluid.layers.resize_bilinear(logit, input.shape[2:])

        if self.enable_auxiliary_loss:
            auxiliary_feat = feat_list[self.backbone_indices[0]]
            auxiliary_logit = self.fcn_head(auxiliary_feat)
            auxiliary_logit = fluid.layers.resize_bilinear(auxiliary_logit, input.shape[2:])

        if self.training:
            loss = model_utils.get_loss(logit, label)
            if self.enable_auxiliary_loss:
                auxiliary_loss = model_utils.get_loss(auxiliary_logit, label)
                loss += (0.4 * auxiliary_loss)
            return loss


        else:
            pred, score_map = model_utils.get_pred_score_map(logit)
            return pred, score_map

    def init_weight(self, pretrained_model=None):
        """
        Initialize the parameters of model parts.

        Args:
            pretrained_model ([str], optional): the pretrained_model path of backbone. Defaults to None.
        """

        if pretrained_model is not None:
            if os.path.exists(pretrained_model):
                utils.load_pretrained_model(self.backbone, pretrained_model)


class PPModule(fluid.dygraph.Layer):
    """
    Pyramid pooling module

    Args:
        in_channels (int): the number of intput channels to pyramid pooling module.

        out_channels (int): the number of output channels after pyramid pooling module.

        bin_sizes (tuple): the out size of pooled feature maps. Default to (1,2,3,6).

        dim_reduction (bool): a bool value represent if reduing dimention after pooling. Default to True.
    """

    def __init__(self, in_channels, out_channels, bin_sizes=(1, 2, 3, 6), dim_reduction=True):
        super(PPModule, self).__init__()
        self.bin_sizes = bin_sizes

        inter_channels = in_channels
        if dim_reduction:
            inter_channels = in_channels // len(bin_sizes)
        
        # we use dimension reduction after pooling mentioned in original implementation.
        self.stages = fluid.dygraph.LayerList([self._make_stage(in_channels, inter_channels, size) for size in bin_sizes])

        self.conv_bn_relu2 = layer_utils.ConvBnRelu(num_channels=in_channels + inter_channels * len(bin_sizes),
                                                    num_filters=out_channels,
                                                    filter_size=3,
                                                    padding=1)

    def _make_stage(self, in_channels, out_channels, size):
        """
        Create one pooling layer.

        In our implementation, we adopt the same dimention reduction as the original paper that might be
        slightly different with other implementations. 

        After pooling, the channels are reduced to 1/len(bin_sizes) immediately, while some other implementations
        keep the channels to be same.


        Args:
            in_channels (int): the number of intput channels to pyramid pooling module.

            size (int): the out size of the pooled layer.

        Returns:
            conv (tensor): a tensor after Pyramid Pooling Module
        """

        # this paddle version does not support AdaptiveAvgPool2d, so skip it here.
        # prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = layer_utils.ConvBnRelu(num_channels=in_channels,
                                      num_filters=out_channels,
                                      filter_size=1)

        return conv

    def forward(self, input):
        cat_layers = []
        for i, stage in enumerate(self.stages):
            size = self.bin_sizes[i]
            x = fluid.layers.adaptive_pool2d(input, pool_size=(size, size), pool_type="max")
            x = stage(x)
            x = fluid.layers.resize_bilinear(x, out_shape=input.shape[2:])
            cat_layers.append(x)
        cat_layers = [input] + cat_layers[::-1]
        cat = fluid.layers.concat(cat_layers, axis=1)
        out = self.conv_bn_relu2(cat)

        return out


@manager.MODELS.add_component
def pspnet_resnet101_vd(*args, **kwargs):
    pretrained_model = None
    return PSPNet(backbone='ResNet101_vd', pretrained_model=pretrained_model, **kwargs)


@manager.MODELS.add_component
def pspnet_resnet101_vd_os8(*args, **kwargs):
    pretrained_model = None
    return PSPNet(backbone='ResNet101_vd', output_stride=8, pretrained_model=pretrained_model, **kwargs)


@manager.MODELS.add_component
def pspnet_resnet50_vd(*args, **kwargs):
    pretrained_model = None
    return PSPNet(backbone='ResNet50_vd', pretrained_model=pretrained_model, **kwargs)


@manager.MODELS.add_component
def pspnet_resnet50_vd_os8(*args, **kwargs):
    pretrained_model = None
    return PSPNet(backbone='ResNet50_vd', output_stride=8, pretrained_model=pretrained_model, **kwargs)

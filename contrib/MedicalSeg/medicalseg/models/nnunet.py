# Implementation of this model is borrowed and modified
# (from torch to paddle) from here:
# https://github.com/MIC-DKFZ/nnUNet

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

import os
import sys
import pickle
import numpy as np
from copy import deepcopy

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from medicalseg.cvlibs import manager
from medicalseg.utils import utils
from medicalseg.cvlibs import param_init
from medicalseg.models import layers

from tools.preprocess_utils import experiment_planner


@manager.MODELS.add_component
class NNUNet(nn.Layer):
    """
    Args:
        plan_path (int): The plan path of nnunet.
        num_classes (int): Only for comparative with other models, this param has no function. Default: 0.
        pretrained (str | optional): The path or url of pretrained model. Default: None.
        stage (int | optional): The stage of nnunet, 0 for nnunet_2d and nnunet_3d, 1 for nnunet_cascade stage 2. Default: None.
        cascade (bool | optional): Whether is cascade model. Default: False.
        deep_supervision (bool | optional): Whether return multi-scale feats when training mode. Default: True.
        feat_map_mul_on_downscale (int | optional): The expansion ratio of stage channels. Defatult: 2.
        use_dropout (bool | optional): Whether use dropout layer in model. Default: False.
        upscale_logits (bool | optional): Whether upscale output feats with different resolutions to the same resolution. Defatult: False.
        convolutional_pooling (bool | optional): Whether add pool layer after conv layer. If convolutional_pooling is True, only conv layer is used and reduce resolution by conv stride. Default: False.
        convolutional_upsampling (bool | optional): Use transpose conv layer or interpolate to upsample feature maps. If True, using transpose conv. Default: False.
    """

    def __init__(
            self,
            plan_path,
            num_classes=0,
            pretrained=None,
            stage=None,
            cascade=False,
            deep_supervision=True,
            feat_map_mul_on_downscale=2,
            use_dropout=False,
            upscale_logits=False,
            convolutional_pooling=True,
            convolutional_upsampling=True, ):
        super().__init__()
        self.plan_path = plan_path
        self.stage = stage
        self.cascade = cascade
        self.load_and_process_plan_file(plan_path)

        if self.threeD:
            conv_op = nn.Conv3D
            dropout_op = nn.Dropout3D
            norm_op = nn.InstanceNorm3D
            max_num_features = experiment_planner.MAX_NUM_FILTERS_3D
        else:
            conv_op = nn.Conv2D
            dropout_op = nn.Dropout2D
            norm_op = nn.InstanceNorm2D
            max_num_features = experiment_planner.MAX_FILTERS_2D

        norm_op_kwargs = {'epsilon': 1e-5}
        dropout_op_kwargs = {'p': 0}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2}
        self.network = Generic_UNet(
            input_channels=self.num_input_channels,
            base_num_features=self.base_num_features,
            num_classes=self.num_classes,
            num_pool=len(self.net_num_pool_op_kernel_sizes),
            num_conv_per_stage=self.conv_per_stage,
            feat_map_mul_on_downscale=feat_map_mul_on_downscale,
            conv_op=conv_op,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=net_nonlin,
            nonlin_kwargs=net_nonlin_kwargs,
            deep_supervision=deep_supervision,
            use_dropout=use_dropout,
            pool_op_kernel_sizes=self.net_num_pool_op_kernel_sizes,
            conv_kernel_sizes=self.net_conv_kernel_sizes,
            upscale_logits=upscale_logits,
            convolutional_pooling=convolutional_pooling,
            convolutional_upsampling=convolutional_upsampling,
            max_num_features=max_num_features)

        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)
        else:
            for sublayer in self.sublayers():
                if isinstance(sublayer,
                              (nn.Conv2D, nn.Conv3D, nn.Conv2DTranspose,
                               nn.Conv3DTranspose)):
                    param_init.kaiming_normal_init(sublayer.weight)
                    if sublayer.bias is not None:
                        param_init.constant_init(sublayer.bias, value=0)

    def load_and_process_plan_file(self, plan_path):
        with open(plan_path, 'rb') as f:
            plans = pickle.load(f)

        if self.stage is None:
            assert len(
                list(plans['plans_per_stage'].keys())
            ) == 1, "If self.stage is None then there can be only one stage in the plans file but got {}. Please specify which stage of the cascade must be trained.".format(
                len(list(plans['plans_per_stage'].keys())))
            self.stage = list(plans['plans_per_stage'].keys())[0]
        self.plans = plans

        stage_plans = self.plans['plans_per_stage'][self.stage]
        self.net_pool_per_axis = stage_plans['num_pool_per_axis']
        self.patch_size = np.array(stage_plans['patch_size']).astype(int)
        self.net_num_pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']
        self.net_conv_kernel_sizes = stage_plans['conv_kernel_sizes']
        self.base_num_features = plans['base_num_features']
        self.num_input_channels = plans['num_modalities']
        self.classes = plans['all_classes']
        self.num_classes = plans['num_classes'] + 1
        if self.stage == 1 and self.cascade:
            self.num_input_channels += (self.num_classes - 1)

        if len(self.patch_size) == 2:
            self.threeD = False
        elif len(self.patch_size) == 3:
            self.threeD = True
        else:
            raise RuntimeError("Invalid patch size in plans file: {}".format(
                self.patch_size))

        if "conv_per_stage" in plans.keys():
            self.conv_per_stage = plans['conv_per_stage']
        else:
            self.conv_per_stage = 2

    def forward(self, x):
        x = self.network(x)
        return x


class StackedConvLayers(nn.Layer):
    def __init__(self,
                 input_feature_channels,
                 output_feature_channels,
                 num_convs,
                 conv_op=nn.Conv2D,
                 conv_kwargs=None,
                 norm_op=nn.BatchNorm2D,
                 norm_op_kwargs=None,
                 dropout_op=nn.Dropout2D,
                 dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU,
                 nonlin_kwargs=None,
                 first_stride=None,
                 basic_block=layers.ConvDropoutNormNonlin):
        super(StackedConvLayers, self).__init__()
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5}
        if conv_kwargs is None:
            conv_kwargs = {
                'kernel_size': 3,
                'stride': 1,
                'padding': 1,
                'dilation': 1,
                'bias_attr': True
            }

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        self.blocks = nn.Sequential(
            basic_block(input_feature_channels, output_feature_channels,
                        self.conv_op, self.conv_kwargs_first_conv, self.norm_op,
                        self.norm_op_kwargs, self.dropout_op,
                        self.dropout_op_kwargs, self.nonlin,
                        self.nonlin_kwargs),
            *[
                basic_block(output_feature_channels, output_feature_channels,
                            self.conv_op, self.conv_kwargs, self.norm_op,
                            self.norm_op_kwargs, self.dropout_op,
                            self.dropout_op_kwargs, self.nonlin,
                            self.nonlin_kwargs) for _ in range(num_convs - 1)
            ])

    def forward(self, x):
        return self.blocks(x)


class Generic_UNet(nn.Layer):
    """
    Args:
        input_channels (int): The input channels of nnUNet.
        base_num_features (int): Basic number of nnUNet channels.
        num_pool (int): The number of MaxPooling.
        num_conv_per_stage (int | optional): The number of conv-bn-nonlin blocks in every stage. Default: 2.
        feat_map_mul_on_downscale (int | optional): The expansion ratio of stage channels. Defatult: 2.
        conv_op (paddle.nn.Layer | optional): The type of conv layer, only support nn.Conv2D and nn.Conv3D. Default: nn.Conv2D.
        norm_op (paddle.nn.Layer | optional): The type of batchnorm layer. Default: nn.BatchNorm2D.
        norm_op_kwwargs (dict | optional): The params for norm_op.
        dropout_op (paddle.nn.Layer | optional): The type of dropout layer. Default: nn.Dropout2D.
        dropout_op_kwargs (dict | optional): The params for dropout_op.
        nonlin (paddle.nn.Layer | optional): The type of activation layer. Default: nn.LeakyReLU.
        nonlin_kwargs (dict | optional): The params for nonlin.
        deep_supervision (bool | optional): Whether return multi-scale feats when training mode. Default: True.
        use_dropout (bool | optional): Whether use dropout layer in model. Default: False.
        pool_op_kernel_sizes (list | optional): The kernel_sizes of pool layers. If None, this param will be computed from num_pool automatically. Default: None.
        conv_kernel_sizes (list | optional): The kernel_sizes of conv layers. If None, this param will be computed from num_pool automatically. Default: None.
        upscale_logits (bool | optional): Whether upscale output feats with different resolutions to the same resolution. Defatult: False.
        convolutional_pooling (bool | optional): Whether add pool layer after conv layer. If convolutional_pooling is True, only conv layer is used and reduce resolution by conv stride. Default: False.
        convolutional_upsampling (bool | optional): Use transpose conv layer or interpolate to upsample feature maps. If True, using transpose conv. Default: False.
        max_num_features (int | optional): The maximum channels of feature maps. Default: None.
        basic_block (paddle.nn.Layer): Only use conv-drop-norm-nonlin module. Default: layers.ConvDropoutNormNonlin.
        seg_output_use_bias (bool | optional): Whether use bias in segmentation head. Default: False.
    """

    def __init__(self,
                 input_channels,
                 base_num_features,
                 num_classes,
                 num_pool,
                 num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2,
                 conv_op=nn.Conv2D,
                 norm_op=nn.BatchNorm2D,
                 norm_op_kwargs=None,
                 dropout_op=nn.Dropout2D,
                 dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU,
                 nonlin_kwargs=None,
                 deep_supervision=True,
                 use_dropout=False,
                 pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False,
                 convolutional_pooling=False,
                 convolutional_upsampling=False,
                 max_num_features=None,
                 basic_block=layers.ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        super().__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'epsilon': 1e-5}
        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias_attr': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self._deep_supervision = deep_supervision

        if conv_op == nn.Conv2D:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2D
            transpconv = nn.Conv2DTranspose
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3D:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3D
            transpconv = nn.Conv3DTranspose
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("Unknown convolution dimensionality, conv op: {}.".
                             format(str(conv_op)))

        self.input_shape_must_be_divisible_by = np.prod(
            pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.max_num_features = max_num_features

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.pool_layers = []
        self.upsample_ops = []
        self.seg_heads = []

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]

            self.conv_blocks_context.append(
                StackedConvLayers(
                    input_features,
                    output_features,
                    num_conv_per_stage,
                    self.conv_op,
                    self.conv_kwargs,
                    self.norm_op,
                    self.norm_op_kwargs,
                    self.dropout_op,
                    self.dropout_op_kwargs,
                    self.nonlin,
                    self.nonlin_kwargs,
                    first_stride,
                    basic_block=basic_block))
            if not self.convolutional_pooling:
                self.pool_layers.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(
                np.round(output_features * feat_map_mul_on_downscale))
            output_features = min(output_features, self.max_num_features)

        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(
            nn.Sequential(
                StackedConvLayers(
                    input_features,
                    output_features,
                    num_conv_per_stage - 1,
                    self.conv_op,
                    self.conv_kwargs,
                    self.norm_op,
                    self.norm_op_kwargs,
                    self.dropout_op,
                    self.dropout_op_kwargs,
                    self.nonlin,
                    self.nonlin_kwargs,
                    first_stride,
                    basic_block=basic_block),
                StackedConvLayers(
                    output_features,
                    final_num_features,
                    1,
                    self.conv_op,
                    self.conv_kwargs,
                    self.norm_op,
                    self.norm_op_kwargs,
                    self.dropout_op,
                    self.dropout_op_kwargs,
                    self.nonlin,
                    self.nonlin_kwargs,
                    basic_block=basic_block)))

        if not use_dropout:
            self.dropout_op_kwargs['p'] = 0.0

        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[-(2 + u
                                                             )].output_channels
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(
                    3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.upsample_ops.append(
                    nn.Upsample(
                        scale_factor=pool_op_kernel_sizes[-(u + 1)],
                        mode=upsample_mode))
            else:
                self.upsample_ops.append(
                    transpconv(
                        nfeatures_from_down,
                        nfeatures_from_skip,
                        pool_op_kernel_sizes[-(u + 1)],
                        pool_op_kernel_sizes[-(u + 1)],
                        bias_attr=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[-(u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[-(u + 1)]
            self.conv_blocks_localization.append(
                nn.Sequential(
                    StackedConvLayers(
                        n_features_after_tu_and_concat,
                        nfeatures_from_skip,
                        num_conv_per_stage - 1,
                        self.conv_op,
                        self.conv_kwargs,
                        self.norm_op,
                        self.norm_op_kwargs,
                        self.dropout_op,
                        self.dropout_op_kwargs,
                        self.nonlin,
                        self.nonlin_kwargs,
                        basic_block=basic_block),
                    StackedConvLayers(
                        nfeatures_from_skip,
                        final_num_features,
                        1,
                        self.conv_op,
                        self.conv_kwargs,
                        self.norm_op,
                        self.norm_op_kwargs,
                        self.dropout_op,
                        self.dropout_op_kwargs,
                        self.nonlin,
                        self.nonlin_kwargs,
                        basic_block=basic_block)))

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_heads.append(
                conv_op(
                    self.conv_blocks_localization[ds][-1].output_channels,
                    num_classes,
                    1,
                    1,
                    0,
                    1,
                    1,
                    bias_attr=seg_output_use_bias))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(
                    nn.Upsample(
                        scale_factor=tuple(
                            [int(i) for i in cum_upsample[usl + 1]]),
                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(layers.Identity())

        self.conv_blocks_localization = nn.LayerList(
            self.conv_blocks_localization)
        self.conv_blocks_context = nn.LayerList(self.conv_blocks_context)
        self.pool_layers = nn.LayerList(self.pool_layers)
        self.upsample_ops = nn.LayerList(self.upsample_ops)
        self.seg_heads = nn.LayerList(self.seg_heads)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.LayerList(self.upscale_logits_ops)

    def forward(self, x):
        skips = []
        outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.pool_layers[d](x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.upsample_ops)):
            x = self.upsample_ops[u](x)
            x = paddle.concat([x, skips[-(u + 1)]], axis=1)
            x = self.conv_blocks_localization[u](x)
            outputs.append(self.seg_heads[u](x))

        if self._deep_supervision and self.training:
            return [[outputs[-1]] + [
                up_op(feat)
                for up_op, feat in zip(
                    list(self.upscale_logits_ops)[::-1], outputs[:-1][::-1])
            ]]
        else:
            return [outputs[-1]]

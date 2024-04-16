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
import paddle.nn as nn
import paddle.nn.functional as F
import paddleseg.utils.logger as logging

from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers
from paddleseg.utils import utils


@manager.MODELS.add_component
class C2FNet(nn.Layer):
    """
     A Coarse-to-Fine Segmentation Network for Small Objects in Remote Sensing Images.

     Args:
         num_classes (int): The unique number of target classes.
         backbone (paddle.nn.Layer): A backbone network.
         backbone_indices (tuple, optional): The values in the tuple indicate the indices of output of backbone.
            Default: (-1, ).
         kernel_sizes(tuple, optional): The sliding windows' size. Default: (128,128).
         training_stride(int, optional): The stride of sliding windows. Default: 32.
         samples_per_gpu(int, optional): The fined process's batch size. Default: 32.
         channels (int, optional): The channels between conv layer and the last layer of FCNHead.
            If None, it will be the number of channels of input features. Default: None.
         align_corners (bool, optional): An argument of `F.interpolate`. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
     """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(-1, ),
                 kernel_sizes=(128, 128),
                 training_stride=32,
                 samples_per_gpu=32,
                 channels=None,
                 align_corners=False,
                 pretrained=None):
        super(C2FNet, self).__init__()
        self.backbone = backbone
        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]
        self.head_fgbg = FCNHead(2, backbone_indices, backbone_channels,
                                 channels)
        self.num_cls = num_classes
        self.kernel_sizes = [kernel_sizes[0], kernel_sizes[1]]
        self.training_stride = training_stride
        self.samples = samples_per_gpu
        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x, heatmaps, label=None):
        ori_heatmap = heatmaps
        heatmap = paddle.argmax(heatmaps, axis=1, keepdim=True, dtype='int32')
        if paddle.max(heatmap) > 15:
            logging.warning(
                "Please note that currently C2FNet can only be trained and evaluated on the iSAID dataset."
            )
        heatmap = paddle.where(
            (heatmap == 10) | (heatmap == 11) | (heatmap == 8) | (heatmap == 15)
            | (heatmap == 9) | (heatmap == 1) | (heatmap == 14),
            paddle.ones_like(heatmap),
            paddle.zeros_like(heatmap)).astype('float32')

        if self.training:
            label = paddle.unsqueeze(label, axis=1).astype('float32')
            label = paddle.where(
                (label == 10) | (label == 11) | (label == 8) | (label == 15) |
                (label == 9) | (label == 1) | (label == 14),
                paddle.ones_like(label), paddle.zeros_like(label))
            mask_regions = F.unfold(heatmap,
                                    kernel_sizes=self.kernel_sizes,
                                    strides=self.training_stride,
                                    paddings=0,
                                    dilations=1,
                                    name=None)
            mask_regions = paddle.transpose(mask_regions, perm=[0, 2, 1])
            mask_regions = paddle.reshape(
                mask_regions,
                shape=[-1, self.kernel_sizes[0] * self.kernel_sizes[1]])

            img_regions = F.unfold(x,
                                   kernel_sizes=self.kernel_sizes,
                                   strides=self.training_stride,
                                   paddings=0,
                                   dilations=1,
                                   name=None)
            img_regions = paddle.transpose(img_regions, perm=[0, 2, 1])
            img_regions = paddle.reshape(
                img_regions,
                shape=[-1, 3 * self.kernel_sizes[0] * self.kernel_sizes[1]])

            label_regions = F.unfold(label,
                                     kernel_sizes=self.kernel_sizes,
                                     strides=self.training_stride,
                                     paddings=0,
                                     dilations=1,
                                     name=None)
            label_regions = paddle.transpose(label_regions, perm=[0, 2, 1])
            label_regions = paddle.reshape(
                label_regions,
                shape=[-1, self.kernel_sizes[0] * self.kernel_sizes[1]])

            mask_regions_sum = paddle.sum(mask_regions, axis=1)
            mask_regions_selected = paddle.where(
                mask_regions_sum > 0, paddle.ones_like(mask_regions_sum),
                paddle.zeros_like(mask_regions_sum))
            final_mask_regions_selected = paddle.zeros_like(
                mask_regions_selected).astype('bool')
            final_mask_regions_selected.stop_gradient = True

            theld = self.samples * x.shape[0]

            if paddle.sum(mask_regions_selected) >= theld:
                _, top_k_idx = paddle.topk(mask_regions_sum, k=theld)
                final_mask_regions_selected[top_k_idx] = True
                selected_img_regions = img_regions[final_mask_regions_selected]
                selected_img_regions = paddle.reshape(selected_img_regions,
                                                      shape=[
                                                          theld, 3,
                                                          self.kernel_sizes[0],
                                                          self.kernel_sizes[1]
                                                      ])

                selected_label_regions = label_regions[
                    final_mask_regions_selected]
                selected_label_regions = paddle.reshape(
                    selected_label_regions,
                    shape=[theld, self.kernel_sizes[0],
                           self.kernel_sizes[1]]).astype('int32')

                feat_list = self.backbone(selected_img_regions)
                bgfg = self.head_fgbg(feat_list)

                binary_fea = F.interpolate(bgfg[0],
                                           self.kernel_sizes,
                                           mode='bilinear',
                                           align_corners=self.align_corners)

                return [binary_fea, selected_label_regions]
            else:
                theld = theld // 8
                _, top_k_idx = paddle.topk(mask_regions_sum, k=theld)
                final_mask_regions_selected[top_k_idx] = True

                selected_img_regions = img_regions[final_mask_regions_selected]
                selected_img_regions = paddle.reshape(selected_img_regions,
                                                      shape=[
                                                          theld, 3,
                                                          self.kernel_sizes[0],
                                                          self.kernel_sizes[1]
                                                      ])

                selected_label_regions = label_regions[
                    final_mask_regions_selected]
                selected_label_regions = paddle.reshape(
                    selected_label_regions,
                    shape=[theld, self.kernel_sizes[0],
                           self.kernel_sizes[1]]).astype('int32')

                feat_list = self.backbone(selected_img_regions)
                bgfg = self.head_fgbg(feat_list)

                binary_fea = F.interpolate(bgfg[0],
                                           self.kernel_sizes,
                                           mode='bilinear',
                                           align_corners=self.align_corners)

                return [binary_fea, selected_label_regions]

        else:
            mask_regions = F.unfold(heatmap,
                                    kernel_sizes=self.kernel_sizes,
                                    strides=self.kernel_sizes[0],
                                    paddings=0,
                                    dilations=1,
                                    name=None)
            mask_regions = paddle.transpose(mask_regions, perm=[0, 2, 1])
            mask_regions = paddle.reshape(
                mask_regions,
                shape=[-1, self.kernel_sizes[0] * self.kernel_sizes[1]])

            img_regions = F.unfold(x,
                                   kernel_sizes=self.kernel_sizes,
                                   strides=self.kernel_sizes[0],
                                   paddings=0,
                                   dilations=1,
                                   name=None)
            img_regions = paddle.transpose(img_regions, perm=[0, 2, 1])
            img_regions = paddle.reshape(
                img_regions,
                shape=[-1, 3 * self.kernel_sizes[0] * self.kernel_sizes[1]])

            mask_regions_sum = paddle.sum(mask_regions, axis=1)
            mask_regions_selected = paddle.where(
                mask_regions_sum > 0, paddle.ones_like(mask_regions_sum),
                paddle.zeros_like(mask_regions_sum)).astype('bool')

            if paddle.sum(mask_regions_selected.astype('int')) == 0:
                return [ori_heatmap]
            else:
                ori_fea_regions = F.unfold(ori_heatmap,
                                           kernel_sizes=self.kernel_sizes,
                                           strides=self.kernel_sizes[0],
                                           paddings=0,
                                           dilations=1,
                                           name=None)
                ori_fea_regions = paddle.transpose(ori_fea_regions,
                                                   perm=[0, 2, 1])
                ori_fea_regions = paddle.reshape(
                    ori_fea_regions,
                    shape=[
                        -1, self.num_cls * self.kernel_sizes[0] *
                        self.kernel_sizes[1]
                    ])
                selected_img_regions = img_regions[mask_regions_selected]
                selected_img_regions = paddle.reshape(
                    selected_img_regions,
                    shape=[
                        selected_img_regions.shape[0], 3, self.kernel_sizes[0],
                        self.kernel_sizes[1]
                    ])
                selected_fea_regions = ori_fea_regions[mask_regions_selected]
                selected_fea_regions = paddle.reshape(
                    selected_fea_regions,
                    shape=[
                        selected_fea_regions.shape[0], self.num_cls,
                        self.kernel_sizes[0], self.kernel_sizes[1]
                    ])
                feat_list = self.backbone(selected_img_regions)
                bgfg = self.head_fgbg(feat_list)
                binary_fea = F.interpolate(bgfg[0],
                                           self.kernel_sizes,
                                           mode='bilinear',
                                           align_corners=self.align_corners)
                binary_fea = F.softmax(binary_fea, axis=1)
                bg_binary, fg_binary = paddle.chunk(binary_fea,
                                                    chunks=2,
                                                    axis=1)
                front, ship, mid, lv, sv, hl, swp, mid2, pl, hb = paddle.split(
                    selected_fea_regions,
                    num_or_sections=[1, 1, 6, 1, 1, 1, 1, 2, 1, 1],
                    axis=1)
                ship = paddle.add(ship, fg_binary)
                lv = paddle.add(lv, fg_binary)
                sv = paddle.add(sv, fg_binary)
                hl = paddle.add(hl, fg_binary)
                swp = paddle.add(swp, fg_binary)
                pl = paddle.add(pl, fg_binary)
                hb = paddle.add(hb, fg_binary)
                selected_fea_regions = paddle.concat(
                    x=[front, ship, mid, lv, sv, hl, swp, mid2, pl, hb], axis=1)
                selected_fea_regions = paddle.reshape(
                    selected_fea_regions,
                    shape=[selected_fea_regions.shape[0], -1])
                ori_fea_regions[mask_regions_selected] = selected_fea_regions
                ori_fea_regions = paddle.reshape(
                    ori_fea_regions,
                    shape=[
                        x.shape[0], -1, self.num_cls * self.kernel_sizes[0] *
                        self.kernel_sizes[1]
                    ])
                ori_fea_regions = paddle.transpose(ori_fea_regions,
                                                   perm=[0, 2, 1])
                fea_out = F.fold(ori_fea_regions, [x.shape[2], x.shape[3]],
                                 self.kernel_sizes,
                                 strides=self.kernel_sizes[0],
                                 paddings=0,
                                 dilations=1,
                                 name=None)

                return [fea_out]

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class FCNHead(nn.Layer):

    def __init__(self,
                 num_classes,
                 backbone_indices=(-1, ),
                 backbone_channels=(270, ),
                 channels=None):
        super(FCNHead, self).__init__()

        self.num_classes = num_classes
        self.backbone_indices = backbone_indices
        if channels is None:
            channels = backbone_channels[0]

        self.conv_1 = layers.ConvBNReLU(in_channels=backbone_channels[0],
                                        out_channels=channels,
                                        kernel_size=1,
                                        stride=1,
                                        bias_attr=True)
        self.cls = nn.Conv2D(in_channels=channels,
                             out_channels=self.num_classes,
                             kernel_size=1,
                             stride=1,
                             bias_attr=True)
        self.init_weight()

    def forward(self, feat_list):
        logit_list = []
        x = feat_list[self.backbone_indices[0]]
        x = self.conv_1(x)
        logit = self.cls(x)
        logit_list.append(logit)
        return logit_list

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)

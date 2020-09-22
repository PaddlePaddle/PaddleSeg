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

import paddle.fluid as fluid
from paddle.fluid.dygraph import Sequential, Conv2D

from paddleseg.cvlibs import manager
from paddleseg.models.common.layer_libs import ConvBNReLU
from paddleseg import utils


class SpatialGatherBlock(fluid.dygraph.Layer):
    def forward(self, pixels, regions):
        n, c, h, w = pixels.shape
        _, k, _, _ = regions.shape

        # pixels: from (n, c, h, w) to (n, h*w, c)
        pixels = fluid.layers.reshape(pixels, (n, c, h * w))
        pixels = fluid.layers.transpose(pixels, (0, 2, 1))

        # regions: from (n, k, h, w) to (n, k, h*w)
        regions = fluid.layers.reshape(regions, (n, k, h * w))
        regions = fluid.layers.softmax(regions, axis=2)

        # feats: from (n, k, c) to (n, c, k, 1)
        feats = fluid.layers.matmul(regions, pixels)
        feats = fluid.layers.transpose(feats, (0, 2, 1))
        feats = fluid.layers.unsqueeze(feats, axes=[-1])

        return feats


class SpatialOCRModule(fluid.dygraph.Layer):
    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 dropout_rate=0.1):
        super(SpatialOCRModule, self).__init__()

        self.attention_block = ObjectAttentionBlock(in_channels, key_channels)
        self.dropout_rate = dropout_rate
        self.conv1x1 = Conv2D(2 * in_channels, out_channels, 1)

    def forward(self, pixels, regions):
        context = self.attention_block(pixels, regions)
        feats = fluid.layers.concat([context, pixels], axis=1)

        feats = self.conv1x1(feats)
        feats = fluid.layers.dropout(feats, self.dropout_rate)

        return feats


class ObjectAttentionBlock(fluid.dygraph.Layer):
    def __init__(self, in_channels, key_channels):
        super(ObjectAttentionBlock, self).__init__()

        self.in_channels = in_channels
        self.key_channels = key_channels

        self.f_pixel = Sequential(
            ConvBNReLU(in_channels, key_channels, 1),
            ConvBNReLU(key_channels, key_channels, 1))

        self.f_object = Sequential(
            ConvBNReLU(in_channels, key_channels, 1),
            ConvBNReLU(key_channels, key_channels, 1))

        self.f_down = ConvBNReLU(in_channels, key_channels, 1)

        self.f_up = ConvBNReLU(key_channels, in_channels, 1)

    def forward(self, x, proxy):
        n, _, h, w = x.shape

        # query : from (n, c1, h1, w1) to (n, h1*w1, key_channels)
        query = self.f_pixel(x)
        query = fluid.layers.reshape(query, (n, self.key_channels, -1))
        query = fluid.layers.transpose(query, (0, 2, 1))

        # key : from (n, c2, h2, w2) to (n, key_channels, h2*w2)
        key = self.f_object(proxy)
        key = fluid.layers.reshape(key, (n, self.key_channels, -1))

        # value : from (n, c2, h2, w2) to (n, h2*w2, key_channels)
        value = self.f_down(proxy)
        value = fluid.layers.reshape(value, (n, self.key_channels, -1))
        value = fluid.layers.transpose(value, (0, 2, 1))

        # sim_map (n, h1*w1, h2*w2)
        sim_map = fluid.layers.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = fluid.layers.softmax(sim_map, axis=-1)

        # context from (n, h1*w1, key_channels) to (n , out_channels, h1, w1)
        context = fluid.layers.matmul(sim_map, value)
        context = fluid.layers.transpose(context, (0, 2, 1))
        context = fluid.layers.reshape(context, (n, self.key_channels, h, w))
        context = self.f_up(context)

        return context


@manager.MODELS.add_component
class OCRNet(fluid.dygraph.Layer):
    def __init__(self,
                 num_classes,
                 backbone,
                 model_pretrained=None,
                 in_channels=None,
                 ocr_mid_channels=512,
                 ocr_key_channels=256,
                 ignore_index=255):
        super(OCRNet, self).__init__()

        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.EPS = 1e-5

        self.backbone = backbone
        self.spatial_gather = SpatialGatherBlock()
        self.spatial_ocr = SpatialOCRModule(ocr_mid_channels, ocr_key_channels,
                                            ocr_mid_channels)
        self.conv3x3_ocr = ConvBNReLU(
            in_channels, ocr_mid_channels, 3, padding=1)
        self.cls_head = Conv2D(ocr_mid_channels, self.num_classes, 1)

        self.aux_head = Sequential(
            ConvBNReLU(in_channels, in_channels, 3, padding=1),
            Conv2D(in_channels, self.num_classes, 1))

        self.init_weight(model_pretrained)

    def forward(self, x, label=None):
        feats = self.backbone(x)

        soft_regions = self.aux_head(feats)
        pixels = self.conv3x3_ocr(feats)

        object_regions = self.spatial_gather(pixels, soft_regions)
        ocr = self.spatial_ocr(pixels, object_regions)

        logit = self.cls_head(ocr)
        logit = fluid.layers.resize_bilinear(logit, x.shape[2:])

        if self.training:
            soft_regions = fluid.layers.resize_bilinear(soft_regions,
                                                        x.shape[2:])
            cls_loss = self._get_loss(logit, label)
            aux_loss = self._get_loss(soft_regions, label)
            return cls_loss + 0.4 * aux_loss

        score_map = fluid.layers.softmax(logit, axis=1)
        score_map = fluid.layers.transpose(score_map, [0, 2, 3, 1])
        pred = fluid.layers.argmax(score_map, axis=3)
        pred = fluid.layers.unsqueeze(pred, axes=[3])
        return pred, score_map

    def init_weight(self, pretrained_model=None):
        """
        Initialize the parameters of model parts.
        Args:
            pretrained_model ([str], optional): the path of pretrained model.. Defaults to None.
        """
        if pretrained_model is not None:
            if os.path.exists(pretrained_model):
                utils.load_pretrained_model(self, pretrained_model)
            else:
                raise Exception('Pretrained model is not found: {}'.format(
                    pretrained_model))

    def _get_loss(self, logit, label):
        """
        compute forward loss of the model

        Args:
            logit (tensor): the logit of model output
            label (tensor): ground truth

        Returns:
            avg_loss (tensor): forward loss
        """
        logit = fluid.layers.transpose(logit, [0, 2, 3, 1])
        label = fluid.layers.transpose(label, [0, 2, 3, 1])
        mask = label != self.ignore_index
        mask = fluid.layers.cast(mask, 'float32')
        loss, probs = fluid.layers.softmax_with_cross_entropy(
            logit,
            label,
            ignore_index=self.ignore_index,
            return_softmax=True,
            axis=-1)

        loss = loss * mask
        avg_loss = fluid.layers.mean(loss) / (
            fluid.layers.mean(mask) + self.EPS)

        label.stop_gradient = True
        mask.stop_gradient = True

        return avg_loss

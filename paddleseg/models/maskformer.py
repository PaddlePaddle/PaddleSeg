# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.models import layers
from paddleseg.cvlibs import manager, param_init
from paddleseg.utils import utils


class BasePixelDecoder(nn.Layer):
    def __init__(self, input_shape, conv_dim=256, norm="GN", mask_dim=256):
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape
                            ]  # starting from "res2" to "res5"
        feature_channels = [v.channels for k, v in input_shape]

        self.lateral_convs, self.output_convs = nn.LayerList(), nn.LayerList()
        use_bias = norm == ''
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                output_conv = layers.ConvNormAct(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    bias_attr=use_bias,
                    norm=nn.GroupNorm(
                        num_groups=32, num_channels=conv_dim),
                    act_type='relu')
                self.output_convs.append(output_conv)
                self.add_sublayer("layer_{}".format(idx + 1), output_conv)
            else:
                lateral_norm = nn.GroupNorm(
                    num_groups=32, num_channels=conv_dim)
                output_norm = nn.GroupNorm(num_groups=32, num_channels=conv_dim)

                lateral_conv = layers.ConvNormAct(
                    in_channels,
                    conv_dim,
                    kernel_size=1,
                    bias_attr=False,
                    norm=lateral_norm)
                output_conv = layers.ConvNormAct(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias_attr=use_bias,
                    norm=output_norm,
                    act_type='relu')
                self.add_sublayer('adapter_{}'.format(idx + 1), lateral_conv)
                self.add_sublayer('layer_{}'.format(idx + 1), output_conv)
                self.lateral_convs.append(lateral_conv)
                self.output_convs.append(output_conv)

        self.lateral_convs = self.lateral_convs[::-1]
        self.output_convs = self.output_convs[::-1]

        self.mask_features = layers.ConvNormAct(
            conv_dim, mask_dim, kernel_size=3, stride=1, padding=1)

    def __init_weight__(self, ):
        for layer in self.sublayers():
            param_init.xavier_uniform(layer.weight)
            if layer.bias is not None:
                param_init.xavier_uniform(layer.bias)

            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)

    def forward(self, features):
        for idx, f in enumerate(self.in_features):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            if lateral_conv is None:
                y = self.output_convs[idx](x)
            else:
                cur_fpn = self.lateral_convs[idx](x)
                y = cur_fpn + F.interpolate(
                    y, size=cur_fpn.shape[-2:], mode='nearest')
                y = self.output_convs[idx](y)
        return self.mask_features(y), None


class PositionEmbeddingSine(nn.Layer):
    def __init__(num_pos_feats=64,
                 temperature=10000,
                 normalize=False,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be true is scale is not None")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = paddle.zeros(
                shape=(x.shape[0], x.shape[2], x.shape[3]), dtype='bool')
        not_mask = ~mask
        y_embed = paddle.cumsum(not_mask, axis=1, dtype='float32')
        x_embed = paddle.cumsum(not_mask, axis=2, dtype='float32')

        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        dim_t = paddle.arange(self.num_pos_feats, dtype='float32')
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = paddle.flatten(
            paddle.stack(
                (paddle.sin(pos_x[:, :, :, 0::2]),
                 paddle.cos(pos_x[:, :, :, 1::2])),
                axis=4),
            start_axis=3)
        pos_y = paddle.flatten(
            paddle.stack((paddle.sin(pos_y[:, :, :, 0::2]),
                          paddle.cos(pos_y[:, :, :, 1::2]))),
            start_axis=3)
        pos = paddle.concat((pos_y, pos_x), axis=3).permute(0, 3, 1, 2)
        return pos


class TransformerPredictor(nn.Layer):
    def __init__(self,
                 in_channels,
                 mask_classification,
                 num_classes=150,
                 hidden_dim=256,
                 num_queries=100,
                 nheads=8,
                 dropout=0.1,
                 dim_feedforward=2048,
                 enc_layers=0,
                 dec_layers=6,
                 pre_norm=False,
                 deep_supervision=True,
                 mask_dim=256,
                 enforce_input_project=False):
        super().__init__()
        self.mask_classification = mask_classification
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision, )

    def forward(self, ):
        pass


class MaskFormerHead(nn.Layer):
    def __init__(self, input_shape, transformer_in_feature='res5'):
        super(MaskFormerHead, self).__init__()
        self.transformer_in_feature = transformer_in_feature
        self.input_shape = input_shape
        self.pixel_decoder = BasePixelDecoder(input_shape)
        self.predictor = TransformerPredictor(
            cfg,
            input_shape[transformer_in_feature].channels,
            mask_classification=True)

    def forward(self, x):  # {"res2": xx, "res3": xx, "res4": xx, "res5": xx}
        mask_features, transformer_encoder_features = self.pixel_decoder(x)
        predictions = self.predictor(x[self.transformer_in_feature],
                                     mask_features)

        return predictions


@manager.MODELS.add_component
class MaskFormer(nn.Layer):
    def __init__(self, backbone, pretrained=None):
        super(MaskFormer, self).__init__()
        self.backbone = backbone
        self.seghead = MaskFormerHead(backbone.output_shape())
        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        features = self.backbone(x)
        outputs = self.sem_seg_head(features)

        if self.training:
            return outputs
        else:
            pass

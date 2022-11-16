# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

from collections import defaultdict
import time

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddleseg
from paddleseg.models import layers
from paddleseg import utils
from paddleseg.cvlibs import manager

from ppmatting.models.losses import MRSD, GradientLoss
from ppmatting.models.backbone import resnet_vd


@manager.MODELS.add_component
class PPMatting(nn.Layer):
    """
    The PPMattinh implementation based on PaddlePaddle.

    The original article refers to
    Guowei Chen, et, al. "PP-Matting: High-Accuracy Natural Image Matting"
    (https://arxiv.org/pdf/2204.09433.pdf).

    Args:
        backbone: backbone model.
        pretrained(str, optional): The path of pretrianed model. Defautl: None.

    """

    def __init__(self, backbone, pretrained=None):
        super().__init__()
        self.backbone = backbone
        self.pretrained = pretrained
        self.loss_func_dict = self.get_loss_func_dict()

        self.backbone_channels = backbone.feat_channels

        self.scb = SCB(self.backbone_channels[-1])

        self.hrdb = HRDB(
            self.backbone_channels[0] + self.backbone_channels[1],
            scb_channels=self.scb.out_channels,
            gf_index=[0, 2, 4])

        self.init_weight()

    def forward(self, inputs):
        x = inputs['img']
        input_shape = paddle.shape(x)
        fea_list = self.backbone(x)

        scb_logits = self.scb(fea_list[-1])
        semantic_map = F.softmax(scb_logits[-1], axis=1)

        fea0 = F.interpolate(
            fea_list[0], input_shape[2:], mode='bilinear', align_corners=False)
        fea1 = F.interpolate(
            fea_list[1], input_shape[2:], mode='bilinear', align_corners=False)
        hrdb_input = paddle.concat([fea0, fea1], 1)
        hrdb_logit = self.hrdb(hrdb_input, scb_logits)
        detail_map = F.sigmoid(hrdb_logit)
        fusion = self.fusion(semantic_map, detail_map)

        if self.training:
            logit_dict = {
                'semantic': semantic_map,
                'detail': detail_map,
                'fusion': fusion
            }
            loss_dict = self.loss(logit_dict, inputs)
            return logit_dict, loss_dict
        else:
            return fusion

    def get_loss_func_dict(self):
        loss_func_dict = defaultdict(list)
        loss_func_dict['semantic'].append(nn.NLLLoss())
        loss_func_dict['detail'].append(MRSD())
        loss_func_dict['detail'].append(GradientLoss())
        loss_func_dict['fusion'].append(MRSD())
        loss_func_dict['fusion'].append(MRSD())
        loss_func_dict['fusion'].append(GradientLoss())
        return loss_func_dict

    def loss(self, logit_dict, label_dict):
        loss = {}

        # semantic loss computation
        # get semantic label
        semantic_label = label_dict['trimap']
        semantic_label_trans = (semantic_label == 128).astype('int64')
        semantic_label_bg = (semantic_label == 0).astype('int64')
        semantic_label = semantic_label_trans + semantic_label_bg * 2
        loss_semantic = self.loss_func_dict['semantic'][0](
            paddle.log(logit_dict['semantic'] + 1e-6),
            semantic_label.squeeze(1))
        loss['semantic'] = loss_semantic

        # detail loss computation
        transparent = label_dict['trimap'] == 128
        detail_alpha_loss = self.loss_func_dict['detail'][0](
            logit_dict['detail'], label_dict['alpha'], transparent)
        # gradient loss
        detail_gradient_loss = self.loss_func_dict['detail'][1](
            logit_dict['detail'], label_dict['alpha'], transparent)
        loss_detail = detail_alpha_loss + detail_gradient_loss
        loss['detail'] = loss_detail
        loss['detail_alpha'] = detail_alpha_loss
        loss['detail_gradient'] = detail_gradient_loss

        # fusion loss
        loss_fusion_func = self.loss_func_dict['fusion']
        # fusion_sigmoid loss
        fusion_alpha_loss = loss_fusion_func[0](logit_dict['fusion'],
                                                label_dict['alpha'])
        # composion loss
        comp_pred = logit_dict['fusion'] * label_dict['fg'] + (
            1 - logit_dict['fusion']) * label_dict['bg']
        comp_gt = label_dict['alpha'] * label_dict['fg'] + (
            1 - label_dict['alpha']) * label_dict['bg']
        fusion_composition_loss = loss_fusion_func[1](comp_pred, comp_gt)
        # grandient loss
        fusion_grad_loss = loss_fusion_func[2](logit_dict['fusion'],
                                               label_dict['alpha'])
        # fusion loss
        loss_fusion = fusion_alpha_loss + fusion_composition_loss + fusion_grad_loss
        loss['fusion'] = loss_fusion
        loss['fusion_alpha'] = fusion_alpha_loss
        loss['fusion_composition'] = fusion_composition_loss
        loss['fusion_gradient'] = fusion_grad_loss

        loss[
            'all'] = 0.25 * loss_semantic + 0.25 * loss_detail + 0.25 * loss_fusion

        return loss

    def fusion(self, semantic_map, detail_map):
        # semantic_map [N, 3, H, W]
        # In index, 0 is foreground, 1 is transition, 2 is backbone
        # After fusion, the foreground is 1, the background is 0, and the transion is between [0, 1]
        index = paddle.argmax(semantic_map, axis=1, keepdim=True)
        transition_mask = (index == 1).astype('float32')
        fg = (index == 0).astype('float32')
        alpha = detail_map * transition_mask + fg
        return alpha

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class SCB(nn.Layer):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = [512 + in_channels, 512, 256, 128, 128, 64]
        self.mid_channels = [512, 256, 128, 128, 64, 64]
        self.out_channels = [256, 128, 64, 64, 64, 3]

        self.psp_module = layers.PPModule(
            in_channels,
            512,
            bin_sizes=(1, 3, 5),
            dim_reduction=False,
            align_corners=False)

        psp_upsamples = [2, 4, 8, 16]
        self.psps = nn.LayerList([
            self.conv_up_psp(512, self.out_channels[i], psp_upsamples[i])
            for i in range(4)
        ])

        scb_list = [
            self._make_stage(
                self.in_channels[i],
                self.mid_channels[i],
                self.out_channels[i],
                padding=int(i == 0) + 1,
                dilation=int(i == 0) + 1)
            for i in range(len(self.in_channels) - 1)
        ]
        scb_list += [
            nn.Sequential(
                layers.ConvBNReLU(
                    self.in_channels[-1], self.mid_channels[-1], 3, padding=1),
                layers.ConvBNReLU(
                    self.mid_channels[-1], self.mid_channels[-1], 3, padding=1),
                nn.Conv2D(
                    self.mid_channels[-1], self.out_channels[-1], 3, padding=1))
        ]
        self.scb_stages = nn.LayerList(scb_list)

    def forward(self, x):
        psp_x = self.psp_module(x)
        psps = [psp(psp_x) for psp in self.psps]

        scb_logits = []
        for i, scb_stage in enumerate(self.scb_stages):
            if i == 0:
                x = scb_stage(paddle.concat((psp_x, x), 1))
            elif i <= len(psps):
                x = scb_stage(paddle.concat((psps[i - 1], x), 1))
            else:
                x = scb_stage(x)
            scb_logits.append(x)
        return scb_logits

    def conv_up_psp(self, in_channels, out_channels, up_sample):
        return nn.Sequential(
            layers.ConvBNReLU(
                in_channels, out_channels, 3, padding=1),
            nn.Upsample(
                scale_factor=up_sample, mode='bilinear', align_corners=False))

    def _make_stage(self,
                    in_channels,
                    mid_channels,
                    out_channels,
                    padding=1,
                    dilation=1):
        layer_list = [
            layers.ConvBNReLU(
                in_channels, mid_channels, 3, padding=1), layers.ConvBNReLU(
                    mid_channels,
                    mid_channels,
                    3,
                    padding=padding,
                    dilation=dilation), layers.ConvBNReLU(
                        mid_channels,
                        out_channels,
                        3,
                        padding=padding,
                        dilation=dilation), nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=False)
        ]
        return nn.Sequential(*layer_list)


class HRDB(nn.Layer):
    """
    The High-Resolution Detail Branch

    Args:
        in_channels(int): The number of input channels.
        scb_channels(list|tuple): The channels of scb logits
        gf_index(list|tuple, optional): Which logit is selected as guidance flow from scb logits. Default: (0, 2, 4)
    """

    def __init__(self, in_channels, scb_channels, gf_index=(0, 2, 4)):
        super().__init__()
        self.gf_index = gf_index
        self.gf_list = nn.LayerList(
            [nn.Conv2D(scb_channels[i], 1, 1) for i in gf_index])

        channels = [64, 32, 16, 8]
        self.res_list = [
            resnet_vd.BasicBlock(
                in_channels, channels[0], stride=1, shortcut=False)
        ]
        self.res_list += [
            resnet_vd.BasicBlock(
                i, i, stride=1) for i in channels[1:-1]
        ]
        self.res_list = nn.LayerList(self.res_list)

        self.convs = nn.LayerList([
            nn.Conv2D(
                channels[i], channels[i + 1], kernel_size=1)
            for i in range(len(channels) - 1)
        ])
        self.gates = nn.LayerList(
            [GatedSpatailConv2d(i, i) for i in channels[1:]])

        self.detail_conv = nn.Conv2D(channels[-1], 1, 1, bias_attr=False)

    def forward(self, x, scb_logits):
        for i in range(len(self.res_list)):
            x = self.res_list[i](x)
            x = self.convs[i](x)
            gf = self.gf_list[i](scb_logits[self.gf_index[i]])
            gf = F.interpolate(
                gf, paddle.shape(x)[-2:], mode='bilinear', align_corners=False)
            x = self.gates[i](x, gf)
        return self.detail_conv(x)


class GatedSpatailConv2d(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias_attr=False):
        super().__init__()
        self._gate_conv = nn.Sequential(
            layers.SyncBatchNorm(in_channels + 1),
            nn.Conv2D(
                in_channels + 1, in_channels + 1, kernel_size=1),
            nn.ReLU(),
            nn.Conv2D(
                in_channels + 1, 1, kernel_size=1),
            layers.SyncBatchNorm(1),
            nn.Sigmoid())
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias_attr)

    def forward(self, input_features, gating_features):
        cat = paddle.concat([input_features, gating_features], axis=1)
        alphas = self._gate_conv(cat)
        x = input_features * (alphas + 1)
        x = self.conv(x)
        return x

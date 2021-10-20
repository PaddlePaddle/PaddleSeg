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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import layers
from paddleseg import utils
from paddleseg.cvlibs import manager

from model import MRSD


def conv_up_psp(in_channels, out_channels, up_sample):
    return nn.Sequential(
        layers.ConvBNReLU(in_channels, out_channels, 3, padding=1),
        nn.Upsample(
            scale_factor=up_sample, mode='bilinear', align_corners=False))


@manager.MODELS.add_component
class ZiYan(nn.Layer):
    def __init__(self, backbone, pretrained=None):
        super().__init__()
        self.backbone = backbone
        self.pretrained = pretrained

        self.backbone_channels = backbone.feat_channels
        ######################
        ### Decoder part - Glance
        ######################
        self.psp_module = layers.PPModule(
            self.backbone_channels[-1],
            512,
            bin_sizes=(1, 3, 5),
            dim_reduction=False,
            align_corners=False)
        self.psp4 = conv_up_psp(512, 256, 2)
        self.psp3 = conv_up_psp(512, 128, 4)
        self.psp2 = conv_up_psp(512, 64, 8)
        self.psp1 = conv_up_psp(512, 64, 16)
        # stage 5g
        self.decoder5_g = nn.Sequential(
            layers.ConvBNReLU(
                512 + self.backbone_channels[-1], 512, 3, padding=1),
            layers.ConvBNReLU(512, 512, 3, padding=2, dilation=2),
            layers.ConvBNReLU(512, 256, 3, padding=2, dilation=2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # stage 4g
        self.decoder4_g = nn.Sequential(
            layers.ConvBNReLU(512, 256, 3, padding=1),
            layers.ConvBNReLU(256, 256, 3, padding=1),
            layers.ConvBNReLU(256, 128, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # stage 3g
        self.decoder3_g = nn.Sequential(
            layers.ConvBNReLU(256, 128, 3, padding=1),
            layers.ConvBNReLU(128, 128, 3, padding=1),
            layers.ConvBNReLU(128, 64, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # stage 2g
        self.decoder2_g = nn.Sequential(
            layers.ConvBNReLU(128, 128, 3, padding=1),
            layers.ConvBNReLU(128, 128, 3, padding=1),
            layers.ConvBNReLU(128, 64, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # stage 1g
        self.decoder1_g = nn.Sequential(
            layers.ConvBNReLU(128, 64, 3, padding=1),
            layers.ConvBNReLU(64, 64, 3, padding=1),
            layers.ConvBNReLU(64, 64, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # stage 0g
        self.decoder0_g = nn.Sequential(
            layers.ConvBNReLU(64, 64, 3, padding=1),
            layers.ConvBNReLU(64, 64, 3, padding=1),
            nn.Conv2D(64, 3, 3, padding=1))

        ##########################
        ### Decoder part - FOCUS
        ##########################
        self.bridge_block = nn.Sequential(
            layers.ConvBNReLU(
                self.backbone_channels[-1], 512, 3, dilation=2, padding=2),
            layers.ConvBNReLU(512, 512, 3, dilation=2, padding=2),
            layers.ConvBNReLU(512, 512, 3, dilation=2, padding=2))
        # stage 5f
        self.decoder5_f = nn.Sequential(
            layers.ConvBNReLU(
                512 + self.backbone_channels[-1], 512, 3, padding=1),
            layers.ConvBNReLU(512, 512, 3, padding=2, dilation=2),
            layers.ConvBNReLU(512, 256, 3, padding=2, dilation=2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # stage 4f
        self.decoder4_f = nn.Sequential(
            layers.ConvBNReLU(
                256 + self.backbone_channels[-2], 256, 3, padding=1),
            layers.ConvBNReLU(256, 256, 3, padding=1),
            layers.ConvBNReLU(256, 128, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # stage 3f
        self.decoder3_f = nn.Sequential(
            layers.ConvBNReLU(
                128 + self.backbone_channels[-3], 128, 3, padding=1),
            layers.ConvBNReLU(128, 128, 3, padding=1),
            layers.ConvBNReLU(128, 64, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # stage 2f
        self.decoder2_f = nn.Sequential(
            layers.ConvBNReLU(
                64 + self.backbone_channels[-4], 128, 3, padding=1),
            layers.ConvBNReLU(128, 128, 3, padding=1),
            layers.ConvBNReLU(128, 64, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # stage 1f
        self.decoder1_f = nn.Sequential(
            layers.ConvBNReLU(
                64 + self.backbone_channels[-5], 64, 3, padding=1),
            layers.ConvBNReLU(64, 64, 3, padding=1),
            layers.ConvBNReLU(64, 64, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # stage 0g
        self.decoder0_f = nn.Sequential(
            layers.ConvBNReLU(64, 64, 3, padding=1),
            layers.ConvBNReLU(64, 64, 3, padding=1),
            nn.Conv2D(64, 1, 3, padding=1))

        self.init_weight()

    def forward(self, inputs):
        x = inputs['img']
        # input fea_list shape [N, 64, H/2, W/2] [N, 64, H/4, W/4]
        # [N, 128, H/8, W/8] [N, 256, H/16, W/16] [N, 512, H/32, W/32]
        fea_list = self.backbone(x)

        ##########################
        ### Decoder part - GLANCE
        ##########################
        #psp: N, 512, H/32, W/32
        psp = self.psp_module(fea_list[-1])
        #d6_g: N, 512, H/16, W/16
        d5_g = self.decoder5_g(paddle.concat((psp, fea_list[-1]), 1))
        #d5_g: N, 512, H/8, W/8
        d4_g = self.decoder4_g(paddle.concat((self.psp4(psp), d5_g), 1))
        #d4_g: N, 256, H/4, W/4
        d3_g = self.decoder3_g(paddle.concat((self.psp3(psp), d4_g), 1))
        #d4_g: N, 128, H/2, W/2
        d2_g = self.decoder2_g(paddle.concat((self.psp2(psp), d3_g), 1))
        #d2_g: N, 64, H, W
        d1_g = self.decoder1_g(paddle.concat((self.psp1(psp), d2_g), 1))
        #d0_g: N, 3, H, W
        d0_g = self.decoder0_g(d1_g)
        # The 1st channel is foreground. The 2nd is transition region. The 3rd is background.
        # glance_sigmoid = F.sigmoid(d0_g)
        glance_sigmoid = F.softmax(d0_g, axis=1)

        ##########################
        ### Decoder part - FOCUS
        ##########################
        bb = self.bridge_block(fea_list[-1])
        #bg: N, 512, H/32, W/32
        d5_f = self.decoder5_f(paddle.concat((bb, fea_list[-1]), 1))
        #d5_f: N, 256, H/16, W/16
        d4_f = self.decoder4_f(paddle.concat((d5_f, fea_list[-2]), 1))
        #d4_f: N, 128, H/8, W/8
        d3_f = self.decoder3_f(paddle.concat((d4_f, fea_list[-3]), 1))
        #d3_f: N, 64, H/4, W/4
        d2_f = self.decoder2_f(paddle.concat((d3_f, fea_list[-4]), 1))
        #d2_f: N, 64, H/2, W/2
        d1_f = self.decoder1_f(paddle.concat((d2_f, fea_list[-5]), 1))
        #d1_f: N, 64, H, W
        d0_f = self.decoder0_f(d1_f)
        #d0_f: N, 1, H, W
        focus_sigmoid = F.sigmoid(d0_f)

        fusion_sigmoid = self.fusion(glance_sigmoid, focus_sigmoid)

        if self.training:
            logit_dict = {
                'glance': glance_sigmoid,
                'focus': focus_sigmoid,
                'fusion': fusion_sigmoid
            }
            return logit_dict
        else:
            return fusion_sigmoid

    def loss(self, logit_dict, label_dict, loss_func_dict=None):
        if loss_func_dict is None:
            loss_func_dict = defaultdict(list)
            loss_func_dict['glance'].append(nn.NLLLoss())
            loss_func_dict['focus'].append(MRSD())
            loss_func_dict['cm'].append(MRSD())
            loss_func_dict['cm'].append(MRSD())

        loss = {}

        # glance loss computation
        # get glance label
        glance_label = label_dict['trimap']
        glance_label_trans = (glance_label == 128).astype('int64')
        glance_label_bg = (glance_label == 0).astype('int64')
        glance_label = glance_label_trans + glance_label_bg * 2
        loss_glance = loss_func_dict['glance'][0](paddle.log(
            logit_dict['glance']), glance_label.squeeze(1))
        loss['glance'] = loss_glance
        # TODO glance label 的验证

        # focus loss computation
        loss_focus = loss_func_dict['focus'][0](logit_dict['focus'],
                                                label_dict['alpha'],
                                                label_dict['trimap'] == 128)
        loss['focus'] = loss_focus

        # collaborative matting loss
        loss_cm_func = loss_func_dict['cm']
        # fusion_sigmoid loss
        loss_cm = loss_cm_func[0](logit_dict['fusion'], label_dict['alpha'])
        # composion loss
        comp_pred = logit_dict['fusion'] * label_dict['fg'] + (
            1 - logit_dict['fusion']) * label_dict['bg']
        loss_cm = loss_cm + loss_cm_func[1](comp_pred, label_dict['img'])
        loss['cm'] = loss_cm

        loss['all'] = 0.25 * loss_glance + 0.25 * loss_focus + 0.25 * loss['cm']

        return loss

    def fusion(self, glance_sigmoid, focus_sigmoid):
        # glance_sigmoid [N, 3, H, W]
        # In index, 0 is foreground, 1 is transition, 2 is backbone
        # After fusion, the foreground is 1, the background is 0, and the transion is between [0, 1]
        index = paddle.argmax(glance_sigmoid, axis=1, keepdim=True)
        transition_mask = (index == 1).astype('float32')
        fg = (index == 0).astype('float32')
        fusion_sigmoid = focus_sigmoid * transition_mask + fg
        return fusion_sigmoid

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


if __name__ == '__main__':
    from resnet_vd import ResNet34_vd
    backbone = ResNet34_vd(output_stride=32)
    x = paddle.randint(0, 256, (1, 3, 320, 320)).astype('float32')
    inputs = {}
    inputs['img'] = x
    model = ZiYan(backbone=backbone)
    logit_dict = model(inputs)
    print(logit_dict)

#     paddle.set_device('cpu')
#     glance_sigmoid = paddle.rand((1, 3, 4, 4))
#     focus_sigmoid = paddle.rand((1, 1, 4, 4))
#     print(glance_sigmoid)
#     print(focus_sigmoid)
#     # fusion(glance_sigmoid, focus_sigmoid)

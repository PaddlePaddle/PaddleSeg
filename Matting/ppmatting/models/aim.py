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

import numpy as np
from skimage.transform import resize

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.cvlibs import manager


from ppmatting.models.losses import AlphaLoss, LaplacianLoss, CompositionLoss
from paddleseg.models.losses import CrossEntropyLoss

@manager.MODELS.add_component
class AIM(nn.Layer):
    """
    The AIM implementation based on PaddlePaddle

    The original article refers to
    Li, Jizhizi, et, al. "Deep Automatic Natural Image Matting"
    (https://arxiv.org/pdf/2107.07235.pdf)

    Args:
        backbone: backbone model.
        pretrained(str, optional): The path of pretrianed model. Defautl: None.
    """
    def __init__(self, backbone, pretrained=None):
        super().__init__()
        self.backbone = backbone
        self.pretrained = pretrained
        self.loss_func_dict = None

        # encoder - resnet
        self.encoder0 = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
        )
        self.mp0 = self.backbone.maxpool1
        self.encoder1 = nn.Sequential(
            self.backbone.layer1
        )
        self.mp1 = self.backbone.maxpool2
        self.encoder2 = self.backbone.layer2
        self.mp2 = self.backbone.maxpool3
        self.encoder3 = self.backbone.layer3
        self.mp3 = self.backbone.maxpool4
        self.encoder4 = self.backbone.layer4
        self.mp4 = self.backbone.maxpool5

        # decoder - global
        self.psp_module = PSPModule(512, 512, (1, 3, 5))
        self.psp4 = conv_up_psp(512, 256, 2)
        self.psp3 = conv_up_psp(512, 128, 4)
        self.psp2 = conv_up_psp(512, 64, 8)
        self.psp1 = conv_up_psp(512, 64, 16)
        self.decoder4_g = nn.Sequential(
            nn.Conv2D(1024, 512, 3, padding=1),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Conv2D(512, 512, 3, padding=1),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Conv2D(512, 256, 3, padding=1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        self.decoder4_g_se = SELayer(256)
        self.decoder3_g = nn.Sequential(
            nn.Conv2D(512, 256, 3, padding=1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2D(256, 256, 3, padding=1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2D(256, 128, 3, padding=1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        self.decoder3_g_se = SELayer(128)
        self.decoder2_g = nn.Sequential(
            nn.Conv2D(256, 128, 3, padding=1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, 128, 3, padding=1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, 64, 3, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        self.decoder2_g_se = SELayer(64)
        self.decoder1_g = nn.Sequential(
            nn.Conv2D(128, 64, 3, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 64, 3, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 64, 3, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        self.decoder1_g_se = SELayer(64)
        self.decoder0_g = nn.Sequential(
            nn.Conv2D(128, 64, 3, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 64, 3, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        self.decoder0_g_spatial = nn.Conv2D(2, 1, 7, padding=3)
        self.decoder0_g_se = SELayer(64)
        self.decoder_final_g = nn.Conv2D(64, 3, 3, padding=1)

        # decoder - local
        self.bridge_block = nn.Sequential(
            nn.Conv2D(512, 512, 3, dilation=2, padding=2),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Conv2D(512, 512, 3, dilation=2, padding=2),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Conv2D(512, 512, 3, dilation=2, padding=2),
            nn.BatchNorm2D(512),
            nn.ReLU())
        self.decoder4_l = nn.Sequential(
            nn.Conv2D(1024, 512, 3, padding=1),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Conv2D(512, 512, 3, padding=1),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Conv2D(512, 256, 3, padding=1),
            nn.BatchNorm2D(256),
            nn.ReLU())
        self.decoder3_l = nn.Sequential(
            nn.Conv2D(512, 256, 3, padding=1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2D(256, 256, 3, padding=1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2D(256, 128, 3, padding=1),
            nn.BatchNorm2D(128),
            nn.ReLU())
        self.decoder2_l = nn.Sequential(
            nn.Conv2D(256, 128, 3, padding=1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, 128, 3, padding=1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, 64, 3, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU())
        self.decoder1_l = nn.Sequential(
            nn.Conv2D(128, 64, 3, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 64, 3, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 64, 3, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU())
        self.decoder0_l = nn.Sequential(
            nn.Conv2D(128, 64, 3, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 64, 3, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU())

        self.decoder_final_l = nn.Conv2D(64, 1, 3, padding=1)

    def _forward(self, src):
        # encoder - modified resnet
        e0 = self.encoder0(src)
        e0p, id0 = self.mp0(e0)
        e1p, id1 = self.mp1(e0p)
        e1 = self.encoder1(e1p)
        e2p, id2 = self.mp2(e1)
        e2 = self.encoder2(e2p)
        e3p, id3 = self.mp3(e2)
        e3 = self.encoder3(e3p)
        e4p, id4 = self.mp4(e3)
        e4 = self.encoder4(e4p)

        # decoder - global: semantic
        psp = self.psp_module(e4)
        d4_g = self.decoder4_g(paddle.concat((psp, e4), 1))
        d4_g = self.decoder4_g_se(d4_g)
        d3_g = self.decoder3_g(paddle.concat((self.psp4(psp), d4_g), 1))
        d3_g = self.decoder3_g_se(d3_g)
        d2_g = self.decoder2_g(paddle.concat((self.psp3(psp), d3_g), 1))
        d2_g = self.decoder2_g_se(d2_g)
        d1_g = self.decoder1_g(paddle.concat((self.psp2(psp), d2_g), 1))
        d1_g = self.decoder1_g_se(d1_g)
        d0_g = self.decoder0_g(paddle.concat((self.psp1(psp), d1_g), 1))
        d0_g_avg = paddle.mean(d0_g, axis=1, keepdim=True)
        d0_g_max = paddle.max(d0_g, axis=1, keepdim=True)
        d0_g_cat = paddle.concat([d0_g_avg, d0_g_max], axis=1)
        d0_g_spatial = self.decoder0_g_spatial(d0_g_cat)
        d0_g_spatial_sigmoid = F.sigmoid(d0_g_spatial)
        d0_g = self.decoder0_g_se(d0_g)
        d0_g = self.decoder_final_g(d0_g)
        global_sigmoid = F.sigmoid(d0_g)

        # decoder - local: matting
        bb = self.bridge_block(e4)
        d4_l = self.decoder4_l(paddle.concat((bb, e4), 1))
        d3_l = F.max_unpool2d(d4_l, id4, kernel_size=2, stride=2)
        d3_l = self.decoder3_l(paddle.concat((d3_l, e3), 1))
        d2_l = F.max_unpool2d(d3_l, id3, kernel_size=2, stride=2)
        d2_l = self.decoder2_l(paddle.concat((d2_l, e2), 1))
        d1_l = F.max_unpool2d(d2_l, id2, kernel_size=2, stride=2)
        d1_l = self.decoder1_l(paddle.concat((d1_l, e1), 1))
        d0_l = F.max_unpool2d(d1_l, id1, kernel_size=2, stride=2)
        d0_l = F.max_unpool2d(d0_l, id0, kernel_size=2, stride=2)
        d0_l = self.decoder0_l(paddle.concat((d0_l, e0), 1))

        d0_l = d0_l + d0_l * d0_g_spatial_sigmoid
        d0_l = self.decoder_final_l(d0_l)
        local_sigmoid = F.sigmoid(d0_l)

        # fusion with global and local
        fusion_sigmoid = self.fusion(
            global_sigmoid, local_sigmoid)

        return global_sigmoid, local_sigmoid, fusion_sigmoid

    def forward(self, data):
        src = data['img']
        _, _, h, w = data['img'].shape

        if self.training:
            global_sigmoid, local_sigmoid, fusion_sigmoid = self._forward(src)
            logit_dict = {
                'local': local_sigmoid,
                'global': global_sigmoid,
                'fusion': fusion_sigmoid
            }
            loss_dict = self.loss(logit_dict, data)
            return logit_dict, loss_dict
        else:
            if data.get("img_g") is None or data.get("img_l") is None:
                _, _, fusion_result = self._forward(src)
                fusion_result = resize(fusion_result, (h, w))
            else:
                src_global, src_local = data['img_g'], data['img_l']
                pred_coutour_g, _, _ = self._forward(src_global)
                pred_coutour_g = pred_coutour_g.cpu().numpy()
                # gen_trimap_from_segmap_e2e
                pred_coutour_g = np.argmax(pred_coutour_g, axis=1)[0].astype(np.int64)
                pred_coutour_g[pred_coutour_g == 1] = 128
                pred_coutour_g[pred_coutour_g == 2] = 255
                pred_coutour_g = pred_coutour_g.astype(np.uint8)

                pred_coutour_g = resize(pred_coutour_g, (h, w)) * 255.0

                _, pred_retouching_l, _ = self._forward(src_local) 
                pred_retouching_l = pred_retouching_l.cpu().numpy()[0, 0, :, :]
                pred_retouching_l = resize(pred_retouching_l, (h, w))

                weighted_global = np.ones(pred_coutour_g.shape)
                weighted_global[pred_coutour_g == 255] = 0
                weighted_global[pred_coutour_g == 0] = 0

                fusion_result = pred_coutour_g * (1. - weighted_global) / 255 + pred_retouching_l * weighted_global
                fusion_result = paddle.to_tensor(fusion_result)
            fusion_result = fusion_result.unsqueeze(0).unsqueeze(0)

            return fusion_result

    def loss(self, logit_dict, label_dict, loss_func_dict=None):
        if loss_func_dict is None:
            if self.loss_func_dict is None:
                self.loss_func_dict = defaultdict(list)
                self.loss_func_dict['local'].append(AlphaLoss())
                self.loss_func_dict['local'].append(LaplacianLoss())
                self.loss_func_dict['global'].append(CrossEntropyLoss())
                self.loss_func_dict['fusion'].append(AlphaLoss())
                self.loss_func_dict['fusion'].append(LaplacianLoss())
                self.loss_func_dict['fusion'].append(CompositionLoss())
        else:
            self.loss_func_dict = loss_func_dict

        loss = {}

        # local loss computation
        loss_local_alpha = self.loss_func_dict['local'][0](logit_dict['local'], label_dict['alpha'], label_dict['trimap'])
        loss_local_laplacian = self.loss_func_dict['local'][1](logit_dict['local'], label_dict['alpha'], label_dict['trimap'])
        loss_local = loss_local_alpha + loss_local_laplacian
        loss['local'] = loss_local

        # global loss computation 
        gt = label_dict['trimap'].clone()
        gt[gt == 0] = 0
        gt[gt == 255] = 2
        gt[gt > 2] = 1
        gt = gt.astype(paddle.float32)
        gt = gt[:, 0, :, :]
        loss_global = self.loss_func_dict['global'][0](logit_dict['global'], gt)
        loss['global'] = loss_global

        # fusion loss computation
        loss_fusion_alpha_whole_img = self.loss_func_dict['fusion'][0](logit_dict['fusion'], label_dict['alpha'])
        loss_fusion_laplacian_whole_img = self.loss_func_dict['fusion'][1](logit_dict['fusion'], label_dict['alpha'])
        loss_fusion_alpha = loss_fusion_alpha_whole_img + loss_fusion_laplacian_whole_img
        loss_fusion_comp = self.loss_func_dict['fusion'][2](logit_dict['fusion'], label_dict['alpha'], label_dict['img'], label_dict['fg'], label_dict['bg'])

        loss_fusion = loss_fusion_alpha + loss_fusion_comp
        loss['fusion'] = loss_fusion

        loss_all = loss_global + loss_local + loss_fusion
        loss['all'] = loss_all

        return loss

    def fusion(self, global_sigmoid, local_sigmoid):
        index = paddle.argmax(global_sigmoid, 1)
        index = index[:, None, :, :].astype(paddle.float32)
        # index <===> [0, 1, 2]
        # bg_mask <===> [1, 0, 0]
        bg_mask = index.clone()
        bg_mask[bg_mask == 2] = 1
        bg_mask = 1 - bg_mask

        # trimap_mask <===> [0, 1, 0]
        trimap_mask = index.clone()
        trimap_mask[trimap_mask == 2] = 0

        # fg_mask <===> [0, 0, 1]
        fg_mask = index.clone()
        fg_mask[fg_mask == 1] = 0
        fg_mask[fg_mask == 2] = 1
        fusion_sigmoid = local_sigmoid * trimap_mask + fg_mask
        
        return fusion_sigmoid


class SELayer(nn.Layer):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias_attr=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias_attr=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).reshape([b, c])
        y = self.fc(y).reshape([b, c, 1, 1])
        return x * y.expand_as(x)


class PSPModule(nn.Layer):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.LayerList(
            [self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2D(
            features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2D(output_size=(size, size))
        conv = nn.Conv2D(features, features, kernel_size=1, bias_attr=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.shape[2], feats.shape[3]
        priors = [F.interpolate(x=stage(feats), size=(
            h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(paddle.concat(priors, 1))
        return self.relu(bottle)


def conv_up_psp(in_channels, out_channels, up_sample):
    return nn.Sequential(
        nn.Conv2D(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2D(out_channels),
        nn.ReLU(),
        nn.Upsample(scale_factor=up_sample, mode='bilinear'))
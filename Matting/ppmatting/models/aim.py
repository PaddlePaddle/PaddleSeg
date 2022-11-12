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


@manager.MODELS.add_component
class AIM(nn.Layer):
    """
    The AIM implementation based on PaddlePaddle.

    The original article refers to
    Jizhizi Li, et, al. "Deep Automatic Natural Image Matting"
    (https://arxiv.org/abs/2107.07235)
    Args:
        backbone: backbone model.
        pretrained(str, optional): The path of pretrianed model. Defautl: None.
    """

    def __init__(self, pretrained=None):
        super(AIM, self).__init__()
        # self.backbone = backbone
        self.pretrained = pretrained

        self.resnet = get_resnet34_mp()
        self.encoder0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )
        self.mp0 = self.resnet.maxpool1
        self.encoder1 = nn.Sequential(
            self.resnet.layer1
        )
        self.mp1 = self.resnet.maxpool2
        self.encoder2 = self.resnet.layer2
        self.mp2 = self.resnet.maxpool3
        self.encoder3 = self.resnet.layer3
        self.mp3 = self.resnet.maxpool4
        self.encoder4 = self.resnet.layer4
        self.mp4 = self.resnet.maxpool5

        self.semantic_decoder = SemanticDecoder()

        # matting decoder
        self.bridge_block = nn.Sequential(
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        self.decoder4_l = Decoder(
            in_channels=1024, out_channels=256, mode="local")
        self.decoder3_l = Decoder(
            in_channels=512, out_channels=128, mode="local")
        self.decoder2_l = Decoder(
            in_channels=256, out_channels=64, mode="local")
        self.decoder1_l = Decoder(
            in_channels=128, out_channels=64, mode="local")
        self.decoder0_l = Decoder(
            in_channels=128, out_channels=64, mode="local")

        self.decoder_final_l = nn.Conv2d(
            in_channels=64, out_channels=1, stride=3, padding=1)

    def forward(self, inputs):
        e0 = self.encoder0(inputs)
        e0p, idx0 = self.mp0(e0)
        e1p, idx1 = self.mp1(e0p)
        e1 = self.encoder1(e1p)
        e2p, idx2 = self.mp2(e1)
        e2 = self.encoder2(e2p)
        e3p, idx3 = self.mp3(e2)
        e3 = self.encoder3(e3p)
        e4p, idx4 = self.mp4(e3)
        e4 = self.encoder4(e4p)

        global_sigmoid, spatial_loss = self.semantic_decoder(e4)

        bb = self.bridge_block(e4)
        d4_l = self.decoder4_l(paddle.concat((bb, e4), 1))
        d3_l = F.max_unpool2d(d4_l, idx4, kernel_size=2, stride=2)
        d3_l = self.decoder3_l(paddle.concat((d3_l, e3), 1))
        d2_l = F.max_unpool2d(d3_l, idx3, kernel_size=2, stride=2)
        d2_l = self.decoder2_l(paddle.concat((d2_l, e2), 1))
        d1_l = F.max_unpool2d(d2_l, idx2, kernel_size=2, stride=2)
        d1_l = self.decoder1_l(paddle.concat((d1_l, e1), 1))
        d0_l = F.max_unpool2d(d1_l, idx1, kernel_size=2, stride=2)
        d0_l = F.max_unpool2d(d0_l, idx0, kernel_size=2, stride=2)
        d0_l = self.decoder0_l(paddle.concat((d0_l, e0), 1))
        d0_l = d0_l + d0_l * spatial_loss
        d0_l = self.decoder_final_l(d0_l)
        local_sigmoid = F.sigmoid(d0_l)

        fusion_sigmoid = get_masked_local_from_global(
            global_sigmoid, local_sigmoid)
        return global_sigmoid, local_sigmoid, fusion_sigmoid


class SELayer(nn.Layer):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(
            nn.Linear(in_features=channel, out_features=channel //
                      reduction, bias_attr=False),
            nn.ReLU(),
            nn.Linear(in_features=channel, out_features=channel //
                      reduction, bias_attr=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).reshape(b, c)
        y = self.fc(y).reshape(b, c, 1, 1)
        return x * y.expand_as(x)


class PSPModule(nn.Layer):
    def __init__(self, in_features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.LayerList(
            [self._make_stage(in_features, size) for size in sizes])
        self.bottleneck = nn.Conv2D(
            in_features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2D(output_size=(size, size))
        conv = nn.Conv2D(features, features, kernel_size=1, bias_attr=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.shape[2], feats.shape[3]
        priors = [F.upsample(x=stage(feats), size=(h, w), mode='bilinear')
                  for stage in self.stages] + [feats]
        bottle = self.bottleneck(paddle.concat(x=priors, axis=1))
        return self.relu(bottle)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")

        self.conv1 = nn.Conv2D(
            inplanes, planes, 3, padding=1, stride=stride, bias_attr=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2D(
            planes, planes, 3, padding=1, stride=stride, bias_attr=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class ResNet34Encoder(nn.Layer):
    def __init__(self,
                 block,
                 layers,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super().__init__()
        # basic parameter
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        # norm layer
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        self._norm_layer = norm_layer

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.conv1 = nn.Conv2D(
            3, self.inplanes, kernel_size=7, stride=1, padding=3, bias_attr=False, weight_attr=nn.initializer.KaimingNormal())
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2D(
            kernel_size=3, stride=2, padding=1, return_mask=True)
        self.maxpool2 = nn.MaxPool2D(
            kernel_size=3, stride=2, padding=1, return_mask=True)
        self.maxpool3 = nn.MaxPool2D(
            kernel_size=3, stride=2, padding=1, return_mask=True)
        self.maxpool4 = nn.MaxPool2D(
            kernel_size=3, stride=2, padding=1, return_mask=True)
        self.maxpool5 = nn.MaxPool2D(
            kernel_size=3, stride=2, padding=1, return_mask=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            in_features=512 * block.expansion, out_features=1000)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    1,
                    stride=stride,
                    bias_attr=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes,
                                planes, groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        x1 = self.conv1(inputs)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1, idx1 = self.maxpool1(x1)

        x2, idx2 = self.maxpool2(x1)
        x2 = self.layer1(x2)

        x3, idx3 = self.maxpool3(x2)
        x3 = self.layer2(x3)

        x4, idx4 = self.maxpool4(x3)
        x4 = self.layer3(x4)

        x5, idx5 = self.maxpool5(x4)
        x5 = self.layer4(x5)

        x_cls = self.avgpool(x5)
        x_cls = torch.flatten(x_cls, 1)
        x_cls = self.fc(x_cls)

        return x_cls


class Decoder(nn.Layer):
    def __init__(self, in_channels, out_channels, mode="global"):
        super().__init__()
        self.conv1 = nn.Conv2D(
            in_channels, in_channels // 2, stride=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.conv2 = nn.Conv2D(
            in_channels // 2, in_channels // 2, stride=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels // 2)
        self.conv3 = nn.Conv2D(
            in_channels // 2, out_channels, stride=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        if mode == "global":
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(self.bn1(x))
        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        x = self.conv3(x)
        x = self.relu(self.bn3(x))

        if self.up is not None:
            x = self.up(x)
        return x


class SemanticDecoder(nn.Layer):
    def __init__(self):
        super(SemanticDecoder, self).__init__()
        self.psp = PSPModule(
            in_features=512, out_features=512, sizes=(1, 3, 5))
        self.psp4 = conv_up_psp(512, 256, 2)
        self.psp3 = conv_up_psp(512, 128, 4)
        self.psp2 = conv_up_psp(512, 64, 8)
        self.psp1 = conv_up_psp(512, 64, 16)

        self.decoder4_g = Decoder(
            in_channels=1024, out_channels=256, mode="global")
        self.decoder4_g_se = SELayer(channel=256)
        self.decoder3_g = Decoder(
            in_channels=512, out_channels=128, mode="global")
        self.decoder3_g_se = SELayer(128)
        self.decoder2_g = Decoder(
            in_channels=256, out_channels=64, mode="global")
        self.decoder2_g_se = SELayer(64)
        self.decoder1_g = Decoder(
            in_channels=128, out_channels=64, mode="global")
        self.decoder1_g_se = SELayer(64)
        self.decoder0_g = Decoder(
            in_channels=128, out_channels=64, mode="global")
        self.decoder0_g_se = SELayer(64)

        self.decoder0_g_spatial = nn.Conv2D(
            in_channels=2, out_channels=1, stride=7, padding=3)
        self.decoder_final_g = nn.Conv2D(
            in_channels=64, out_channels=3, stride=3, padding=1)

    def forward(self, inputs):
        """
        Args:
            inputs (paddle.Tensor): feature from encoder

        Returns:
            global_sigmoid: Semantic Loss
            d0_g_spatial_sigmoid: spatial loss
        """
        psp = self.psp_module(inputs)
        d4_g = self.decoder4_g(paddle.concat((psp, inputs), 1))
        d4_g = self.decoder4_g_se(d4_g)
        d3_g = self.decoder3_g(paddle.concat((self.psp4(psp), d4_g), 1))
        d3_g = self.decoder3_g_se(d3_g)
        d2_g = self.decoder2_g(paddle.concat((self.psp3(psp), d3_g), 1))
        d2_g = self.decoder2_g_se(d2_g)
        d1_g = self.decoder1_g(paddle.concat((self.psp2(psp), d2_g), 1))
        d1_g = self.decoder1_g_se(d1_g)
        d0_g = self.decoder0_g(paddle.concat((self.psp1(psp), d1_g), 1))

        d0_g_avg = paddle.mean(d0_g, dim=1, keepdim=True)
        d0_g_max = paddle.max(d0_g, dim=1, keepdim=True)
        d0_g_cat = paddle.concat([d0_g_avg, d0_g_max], axis=1)

        d0_g_spatial = self.decoder0_g_spatial(d0_g_cat)
        d0_g_spatial_sigmoid = F.sigmoid(d0_g_spatial)
        d0_g = self.decoder0_g_se(d0_g)
        d0_g = self.decoder_final_g(d0_g)
        global_sigmoid = F.sigmoid(d0_g)

        return global_sigmoid, d0_g_spatial_sigmoid


def get_masked_local_from_global(global_sigmoid, local_sigmoid):
    values, index = paddle.max(global_sigmoid, 1)
    index = index[:, None, :, :].float()
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


def conv_up_psp(in_channels, out_channels, up_sample):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=up_sample, mode='bilinear'))

def get_resnet34_mp(**kwargs):
    model = ResNet34Encoder(BasicBlock, [3, 4, 6, 3], **kwargs)
    # checkpoint = paddle.load("models/pretrained/r34mp_pretrained_imagenet.pth.tar")
    # model.load_state_dict(checkpoint)
    return model
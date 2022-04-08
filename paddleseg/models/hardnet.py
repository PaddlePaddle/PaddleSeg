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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils


@manager.MODELS.add_component
class HarDNet(nn.Layer):
    """
    [Real Time] The FC-HardDNet 70 implementation based on PaddlePaddle.
    The original article refers to
        Chao, Ping, et al. "HarDNet: A Low Memory Traffic Network"
        (https://arxiv.org/pdf/1909.00948.pdf)

    Args:
        num_classes (int): The unique number of target classes.
        stem_channels (tuple|list, optional): The number of channels before the encoder. Default: (16, 24, 32, 48).
        ch_list (tuple|list, optional): The number of channels at each block in the encoder. Default: (64, 96, 160, 224, 320).
        grmul (float, optional): The channel multiplying factor in HarDBlock, which is m in the paper. Default: 1.7.
        gr (tuple|list, optional): The growth rate in each HarDBlock, which is k in the paper. Default: (10, 16, 18, 24, 32).
        n_layers (tuple|list, optional): The number of layers in each HarDBlock. Default: (4, 4, 8, 8, 8).
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 stem_channels=(16, 24, 32, 48),
                 ch_list=(64, 96, 160, 224, 320),
                 grmul=1.7,
                 gr=(10, 16, 18, 24, 32),
                 n_layers=(4, 4, 8, 8, 8),
                 align_corners=False,
                 pretrained=None):

        super().__init__()
        self.align_corners = align_corners
        self.pretrained = pretrained
        encoder_blks_num = len(n_layers)
        decoder_blks_num = encoder_blks_num - 1
        encoder_in_channels = stem_channels[3]

        self.stem = nn.Sequential(
            layers.ConvBNReLU(
                3, stem_channels[0], kernel_size=3, bias_attr=False),
            layers.ConvBNReLU(
                stem_channels[0],
                stem_channels[1],
                kernel_size=3,
                bias_attr=False),
            layers.ConvBNReLU(
                stem_channels[1],
                stem_channels[2],
                kernel_size=3,
                stride=2,
                bias_attr=False),
            layers.ConvBNReLU(
                stem_channels[2],
                stem_channels[3],
                kernel_size=3,
                bias_attr=False))

        self.encoder = Encoder(encoder_blks_num, encoder_in_channels, ch_list,
                               gr, grmul, n_layers)

        skip_connection_channels = self.encoder.get_skip_channels()
        decoder_in_channels = self.encoder.get_out_channels()

        self.decoder = Decoder(decoder_blks_num, decoder_in_channels,
                               skip_connection_channels, gr, grmul, n_layers,
                               align_corners)

        self.cls_head = nn.Conv2D(
            in_channels=self.decoder.get_out_channels(),
            out_channels=num_classes,
            kernel_size=1)

        self.init_weight()

    def forward(self, x):
        input_shape = paddle.shape(x)[2:]
        x = self.stem(x)
        x, skip_connections = self.encoder(x)
        x = self.decoder(x, skip_connections)
        logit = self.cls_head(x)
        logit = F.interpolate(
            logit,
            size=input_shape,
            mode="bilinear",
            align_corners=self.align_corners)
        return [logit]

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class Encoder(nn.Layer):
    """The Encoder implementation of FC-HardDNet 70.

    Args:
        n_blocks (int): The number of blocks in the Encoder module.
        in_channels (int): The number of input channels.
        ch_list (tuple|list): The number of channels at each block in the encoder.
        grmul (float): The channel multiplying factor in HarDBlock, which is m in the paper.
        gr (tuple|list): The growth rate in each HarDBlock, which is k in the paper.
        n_layers (tuple|list): The number of layers in each HarDBlock.
    """

    def __init__(self, n_blocks, in_channels, ch_list, gr, grmul, n_layers):
        super().__init__()
        self.skip_connection_channels = []
        self.shortcut_layers = []
        self.blks = nn.LayerList()
        ch = in_channels
        for i in range(n_blocks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i])
            ch = blk.get_out_ch()
            self.skip_connection_channels.append(ch)
            self.blks.append(blk)
            if i < n_blocks - 1:
                self.shortcut_layers.append(len(self.blks) - 1)
            self.blks.append(
                layers.ConvBNReLU(
                    ch, ch_list[i], kernel_size=1, bias_attr=False))

            ch = ch_list[i]
            if i < n_blocks - 1:
                self.blks.append(nn.AvgPool2D(kernel_size=2, stride=2))
        self.out_channels = ch

    def forward(self, x):
        skip_connections = []
        for i in range(len(self.blks)):
            x = self.blks[i](x)
            if i in self.shortcut_layers:
                skip_connections.append(x)
        return x, skip_connections

    def get_skip_channels(self):
        return self.skip_connection_channels

    def get_out_channels(self):
        return self.out_channels


class Decoder(nn.Layer):
    """The Decoder implementation of FC-HardDNet 70.

    Args:
        n_blocks (int): The number of blocks in the Encoder module.
        in_channels (int): The number of input channels.
        skip_connection_channels (tuple|list): The channels of shortcut layers in encoder.
        grmul (float): The channel multiplying factor in HarDBlock, which is m in the paper.
        gr (tuple|list): The growth rate in each HarDBlock, which is k in the paper.
        n_layers (tuple|list): The number of layers in each HarDBlock.
    """

    def __init__(self,
                 n_blocks,
                 in_channels,
                 skip_connection_channels,
                 gr,
                 grmul,
                 n_layers,
                 align_corners=False):
        super().__init__()
        prev_block_channels = in_channels
        self.n_blocks = n_blocks
        self.dense_blocks_up = nn.LayerList()
        self.conv1x1_up = nn.LayerList()

        for i in range(n_blocks - 1, -1, -1):
            cur_channels_count = prev_block_channels + skip_connection_channels[
                i]
            conv1x1 = layers.ConvBNReLU(
                cur_channels_count,
                cur_channels_count // 2,
                kernel_size=1,
                bias_attr=False)
            blk = HarDBlock(
                base_channels=cur_channels_count // 2,
                growth_rate=gr[i],
                grmul=grmul,
                n_layers=n_layers[i])

            self.conv1x1_up.append(conv1x1)
            self.dense_blocks_up.append(blk)

            prev_block_channels = blk.get_out_ch()

        self.out_channels = prev_block_channels
        self.align_corners = align_corners

    def forward(self, x, skip_connections):
        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            x = F.interpolate(
                x,
                size=paddle.shape(skip)[2:],
                mode="bilinear",
                align_corners=self.align_corners)
            x = paddle.concat([x, skip], axis=1)
            x = self.conv1x1_up[i](x)
            x = self.dense_blocks_up[i](x)
        return x

    def get_out_channels(self):
        return self.out_channels


class HarDBlock(nn.Layer):
    """The HarDBlock implementation

    Args:
        base_channels (int): The base channels.
        growth_rate (tuple|list): The growth rate.
        grmul (float): The channel multiplying factor.
        n_layers (tuple|list): The number of layers.
        keepBase (bool, optional): A bool value indicates whether concatenating the first layer. Default: False.
    """

    def __init__(self,
                 base_channels,
                 growth_rate,
                 grmul,
                 n_layers,
                 keepBase=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0
        for i in range(n_layers):
            outch, inch, link = get_link(i + 1, base_channels, growth_rate,
                                         grmul)

            self.links.append(link)
            layers_.append(
                layers.ConvBNReLU(
                    inch, outch, kernel_size=3, bias_attr=False))
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        self.layers = nn.LayerList(layers_)

    def forward(self, x):
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = paddle.concat(tin, axis=1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)

        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or \
                (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = paddle.concat(out_, 1)

        return out

    def get_out_ch(self):
        return self.out_channels


def get_link(layer, base_ch, growth_rate, grmul):
    if layer == 0:
        return base_ch, 0, []
    out_channels = growth_rate
    link = []
    for i in range(10):
        dv = 2**i
        if layer % dv == 0:
            k = layer - dv
            link.insert(0, k)
            if i > 0:
                out_channels *= grmul
    out_channels = int(int(out_channels + 1) / 2) * 2
    in_channels = 0
    for i in link:
        ch, _, _ = get_link(i, base_ch, growth_rate, grmul)
        in_channels += ch
    return out_channels, in_channels, link

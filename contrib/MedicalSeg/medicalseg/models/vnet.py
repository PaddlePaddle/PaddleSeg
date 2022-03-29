# Implementation of this model is borrowed and modified
# (from torch to paddle) from here:
# https://github.com/black0017/MedicalZooPytorch
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

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from medicalseg.cvlibs import manager
from medicalseg.utils import utils


class LUConv(nn.Layer):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = nn.ELU() if elu else nn.PReLU(nchan)
        self.conv1 = nn.Conv3D(nchan, nchan, kernel_size=5, padding=2)

        self.bn1 = nn.BatchNorm3D(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))

        return out


def _make_nConv(nchan, depth, elu):
    """
    Make depth number of layer(convbnrelu) and don't change the channel
    Add Nonlinearity into the network
    """
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Layer):
    """
    Transfer the input into 16 channels + tiled input
    """

    def __init__(self, in_channels, elu):
        super(InputTransition, self).__init__()
        self.num_features = 16
        self.in_channels = in_channels

        self.conv1 = nn.Conv3D(
            self.in_channels, self.num_features, kernel_size=5, padding=2)

        self.bn1 = nn.BatchNorm3D(self.num_features)

        self.relu1 = nn.ELU() if elu else nn.PReLU(self.num_features)

    def forward(self, x):
        out = self.conv1(x)
        repeat_rate = int(self.num_features / self.in_channels)
        out = self.bn1(out)
        x_tile = x.tile([1, repeat_rate, 1, 1, 1])
        return self.relu1(paddle.add(out, x_tile))


class DownTransition(nn.Layer):
    def __init__(self,
                 inChans,
                 nConvs,
                 elu,
                 dropout=False,
                 downsample_stride=(2, 2, 2),
                 kernel=(2, 2, 2)):
        """
        1. double the output channel and downsample the input using down_conv(the kernel size can be changed)
        2. add dropout by option
        3. add nConvs layer to add linearity and add with original downsample one
        """
        super(DownTransition, self).__init__()
        outChans = 2 * inChans
        self.if_dropout = dropout
        self.down_conv = nn.Conv3D(
            inChans, outChans, kernel_size=kernel, stride=downsample_stride)
        self.bn1 = nn.BatchNorm3D(outChans)
        self.relu1 = nn.ELU() if elu else nn.PReLU(outChans)
        self.relu2 = nn.ELU() if elu else nn.PReLU(outChans)
        self.dropout = nn.Dropout3D()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.dropout(down) if self.if_dropout else down
        out = self.ops(out)
        out = paddle.add(out, down)
        out = self.relu2(out)

        return out


class UpTransition(nn.Layer):
    def __init__(self,
                 inChans,
                 outChans,
                 nConvs,
                 elu,
                 dropout=False,
                 dropout2=False,
                 upsample_stride_size=(2, 2, 2),
                 kernel=(2, 2, 2)):
        super(UpTransition, self).__init__()
        """
        1. Add dropout to input and skip input optionally (generalization)
        2. Use Conv3DTranspose to upsample (upsample)
        3. concate the upsampled and skipx (multi-leval feature fusion)
        4. Add nConvs convs and residually add with result of step(residual + nonlinearity)
        """
        self.up_conv = nn.Conv3DTranspose(
            inChans,
            outChans // 2,
            kernel_size=kernel,
            stride=upsample_stride_size)

        self.bn1 = nn.BatchNorm3D(outChans // 2)
        self.relu1 = nn.ELU() if elu else nn.PReLU(outChans // 2)
        self.relu2 = nn.ELU() if elu else nn.PReLU(outChans)
        self.if_dropout = dropout
        self.if_dropout2 = dropout2
        self.dropout1 = nn.Dropout3D()
        self.dropout2 = nn.Dropout3D()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.dropout1(x) if self.if_dropout else x
        skipx = self.dropout2(skipx) if self.if_dropout2 else skipx
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = paddle.concat((out, skipx), 1)
        out = self.ops(xcat)
        out = self.relu2(paddle.add(out, xcat))

        return out


class OutputTransition(nn.Layer):
    def __init__(self, in_channels, num_classes, elu):
        """
        conv the output down to channels as the desired classesv
        """
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3D(
            in_channels, num_classes, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3D(num_classes)

        self.conv2 = nn.Conv3D(num_classes, num_classes, kernel_size=1)
        self.relu1 = nn.ELU() if elu else nn.PReLU(num_classes)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return out


@manager.MODELS.add_component
class VNet(nn.Layer):
    """
    Implementations based on the Vnet paper: https://arxiv.org/abs/1606.04797
    """

    def __init__(self,
                 elu=False,
                 in_channels=1,
                 num_classes=4,
                 pretrained=None,
                 kernel_size=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
                 stride_size=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2))):
        super().__init__()
        self.best_loss = 1000000
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.in_tr = InputTransition(in_channels, elu=elu)
        self.down_tr32 = DownTransition(
            16, 1, elu, downsample_stride=stride_size[0], kernel=kernel_size[0])
        self.down_tr64 = DownTransition(
            32, 2, elu, downsample_stride=stride_size[1], kernel=kernel_size[1])
        self.down_tr128 = DownTransition(
            64,
            3,
            elu,
            dropout=True,
            downsample_stride=stride_size[2],
            kernel=kernel_size[2])
        self.down_tr256 = DownTransition(
            128,
            2,
            elu,
            dropout=True,
            downsample_stride=stride_size[3],
            kernel=kernel_size[3])
        self.up_tr256 = UpTransition(
            256,
            256,
            2,
            elu,
            dropout=True,
            dropout2=True,
            upsample_stride_size=stride_size[3],
            kernel=kernel_size[3])
        self.up_tr128 = UpTransition(
            256,
            128,
            2,
            elu,
            dropout=True,
            dropout2=True,
            upsample_stride_size=stride_size[2],
            kernel=kernel_size[2])
        self.up_tr64 = UpTransition(
            128,
            64,
            1,
            elu,
            upsample_stride_size=stride_size[1],
            kernel=kernel_size[1])
        self.up_tr32 = UpTransition(
            64,
            32,
            1,
            elu,
            upsample_stride_size=stride_size[0],
            kernel=kernel_size[0])
        self.out_tr = OutputTransition(32, num_classes, elu)

        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        out16 = self.in_tr(x)  # dropout cause a lot align problem
        out32 = self.down_tr32(out16)  # [4, 32, 256, 256, 9]
        out64 = self.down_tr64(out32)  # [4, 64, 128, 128, 8]
        out128 = self.down_tr128(out64)  # [4, 128, 64, 64, 4]
        out256 = self.down_tr256(out128)  # [4, 256, 32, 32, 2]
        out = self.up_tr256(out256, out128)  # [4, 256, 64, 64, 4]
        out = self.up_tr128(out, out64)  # [4, 128, 128, 128, 8]
        out = self.up_tr64(out, out32)  # [4, 64, 256, 256, 9]
        out = self.up_tr32(out, out16)  # [4, 32, 512, 512, 12]
        out = self.out_tr(out)
        return [out, ]

    def test(self):
        import numpy as np
        np.random.seed(1)
        a = np.random.rand(1, self.in_channels, 32, 32, 32)
        input_tensor = paddle.to_tensor(a, dtype='float32')

        ideal_out = paddle.rand((1, self.num_classes, 32, 32, 32))
        out = self.forward(input_tensor)[0]
        print("out", out.mean(), input_tensor.mean())

        assert ideal_out.shape == out.shape
        paddle.summary(self, (1, self.in_channels, 32, 32, 32))

        print("Vnet test is complete")


if __name__ == "__main__":
    from reprod_log import ReprodLogger, ReprodDiffHelper
    import numpy as np
    diff_helper = ReprodDiffHelper()

    torch_info = diff_helper.load_info("../../data/vnet_align/train_ref.npy")
    paddle_info = diff_helper.load_info(
        "../../data/vnet_align/train_paddle.npy")
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="../../data/vnet_align/train_diff.log")

    # m = VNet(in_channels=1, num_classes=3)
    # # paddle.save(m.state_dict(), "../../data/vnet_align/paddlemodelorg.pdparams")
    # m.set_dict(paddle.load("../../data/vnet_align/paddlemodel.pdparams"))
    # m.eval()

    # x = paddle.to_tensor(np.load("../../data/vnet_align/fake_data.npy"), dtype="float32")
    # y = paddle.to_tensor(np.load("../../data/vnet_align/fake_label.npy"), dtype="int64")
    # x.stop_gradient = False
    # out = m(x)[0]
    # print(out.mean())

    # test forward
    # reprod_logger = ReprodLogger()
    # reprod_logger.add("x", x.cpu().detach().numpy())
    # reprod_logger.add("y", y.cpu().detach().numpy())
    # reprod_logger.add("logits", out.cpu().detach().numpy())
    # reprod_logger.save("../../data/vnet_align/forward_paddle.npy")

    # # implement loss
    # from medicalseg.models.losses import CrossEntropyLoss, DiceLoss
    # criterion = [DiceLoss(), CrossEntropyLoss()]

    # losses = None
    # reprod_logger = ReprodLogger()
    # for i, loss in enumerate(criterion):
    #     if type(loss).__name__ == 'DiceLoss':
    #         loss_val, per_channel_dice = loss(out, y)
    #     else:

    #         loss_val = loss(out, y)

    #     if losses is None:
    #         losses = loss_val
    #     else:
    #         losses += loss_val

    #     reprod_logger.add("loss_{}".format(i), loss_val.cpu().detach().numpy())
    # print("loss_{}".format(i), loss_val, loss_val.cpu().detach().numpy())

    # reprod_logger.save("../../data/vnet_align/losses_paddle.npy")

    # # compare forward result and produce log
    # diff_helper = ReprodDiffHelper()
    # torch_info = diff_helper.load_info("../../data/vnet_align/forward_torch.npy")
    # paddle_info = diff_helper.load_info("../../data/vnet_align/forward_paddle.npy")
    # diff_helper.compare_info(torch_info, paddle_info)
    # diff_helper.report(path="../../data/vnet_align/forward_diff.log")

    # # compare loss result and produce log
    # diff_helper = ReprodDiffHelper()
    # torch_info = diff_helper.load_info("../../data/vnet_align/losses_torch.npy")
    # paddle_info = diff_helper.load_info("../../data/vnet_align/losses_paddle.npy")
    # diff_helper.compare_info(torch_info, paddle_info)
    # diff_helper.report(path="../../data/vnet_align/losses_diff.log")

    ###### test for iters ##########
    # from medicalseg.models.losses import CrossEntropyLoss, DiceLoss
    # reprod_logger = ReprodLogger()
    # # init optimizer
    # max_iter = 5
    # lr = 1e-3
    # momentum = 0.9

    # m = VNet(in_channels=1, num_classes=3)
    # m.set_dict(paddle.load("../../data/vnet_align/paddlemodel.pdparams"))
    # m.eval()

    # criterion = [DiceLoss(), CrossEntropyLoss()]
    # opt_paddle = paddle.optimizer.Momentum(
    #     learning_rate=lr,
    #     momentum=momentum,
    #     parameters=m.parameters())

    # x = paddle.to_tensor(np.load("../../data/vnet_align/fake_data.npy"), dtype="float32")
    # y = paddle.to_tensor(np.load("../../data/vnet_align/fake_label.npy"), dtype="int64")
    # x.stop_gradient = False

    # for idx in range(max_iter):
    #     out = m(x)[0]

    #     # compute loss
    #     losses = None
    #     for i, loss in enumerate(criterion):
    #         if type(loss).__name__ == 'DiceLoss':
    #             loss_val, per_channel_dice = loss(out, y)
    #         else:

    #             loss_val = loss(out, y)

    #         if losses is None:
    #             losses = loss_val
    #         else:
    #             losses += loss_val

    #     reprod_logger.add("loss_{}".format(idx), losses.cpu().detach().numpy())
    #     print("loss_{}".format(idx), losses)

    #     opt_paddle.clear_grad()
    #     losses.backward()
    #     opt_paddle.step()

    # reprod_logger.save("../../data/vnet_align/backward_paddle.npy")

    # # # compare loss result and produce log
    # diff_helper = ReprodDiffHelper()
    # torch_info = diff_helper.load_info("../../data/vnet_align/backward_ref.npy")
    # paddle_info = diff_helper.load_info("../../data/vnet_align/backward_paddle.npy")
    # diff_helper.compare_info(torch_info, paddle_info)
    # diff_helper.report(path="../../data/vnet_align/backward_diff.log")

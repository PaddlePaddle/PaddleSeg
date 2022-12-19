import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg import utils
from paddleseg.models.backbones.transformer_utils import Identity, DropPath


class InjectionMultiSumCBR(nn.Layer):
    def __init__(self, in_channels, out_channels, activations=None):
        '''
        local_embedding: conv-bn-relu
        global_embedding: conv-bn-relu
        global_act: conv
        '''
        super(InjectionMultiSumCBR, self).__init__()

        self.local_embedding = ConvBNAct(
            in_channels, out_channels, kernel_size=1)
        self.global_embedding = ConvBNAct(
            in_channels, out_channels, kernel_size=1)
        self.global_act = ConvBNAct(
            in_channels, out_channels, kernel_size=1, norm=None, act=None)
        self.act = HSigmoid()

    def forward(self, x_low, x_global):
        xl_hw = paddle.shape(x)[2:]
        local_feat = self.local_embedding(x_low)
        # kernel
        global_act = self.global_act(x_global)
        global_act = F.interpolate(
            self.act(global_act), xl_hw, mode='bilinear', align_corners=False)
        # feat_h
        global_feat = self.global_embedding(x_global)
        global_feat = F.interpolate(
            global_feat, xl_hw, mode='bilinear', align_corners=False)
        out = local_feat * global_act + global_feat
        return out


class FuseBlockSum(nn.Layer):
    def __init__(self, in_channels, out_channels, activations=None):
        super(FuseBlockSum, self).__init__()

        self.fuse1 = ConvBNAct(
            in_channels, out_channels, kernel_size=1, act=None)
        self.fuse2 = ConvBNAct(
            in_channels, out_channels, kernel_size=1, act=None)

    def forward(self, x_low, x_high):
        xl_hw = paddle.shape(x)[2:]
        inp = self.fuse1(x_low)
        kernel = self.fuse2(x_high)
        feat_h = F.interpolate(
            kernel, xl_hw, mode='bilinear', align_corners=False)
        out = inp + feat_h
        return out


class FuseBlockMulti(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1,
            activations=None, ):
        super(FuseBlockMulti, self).__init__()
        assert stride in [1, 2], "The stride should be 1 or 2."

        self.fuse1 = ConvBNAct(
            in_channels, out_channels, kernel_size=1, act=None)
        self.fuse2 = ConvBNAct(
            in_channels, out_channels, kernel_size=1, act=None)
        self.act = HSigmoid()

    def forward(self, x_low, x_high):
        xl_hw = paddle.shape(x)[2:]
        inp = self.fuse1(x_low)
        sig_act = self.fuse2(x_high)
        sig_act = F.interpolate(
            self.act(sig_act), xl_hw, mode='bilinear', align_corners=False)
        out = inp * sig_act
        return out


class InjectionMultiSumSimple(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 activations=None,
                 global_in_channels=None,
                 lr_mult=1.0):
        super(InjectionMultiSumSimple, self).__init__()

        self.local_embedding = ConvBNAct(
            in_channels, out_channels, kernel_size=1, lr_mult=lr_mult)
        self.global_embedding = ConvBNAct(
            global_in_channels, out_channels, kernel_size=1, lr_mult=lr_mult)
        # self.global_act = ConvBNAct(
        #     global_in_channels, out_channels, kernel_size=1, lr_mult=lr_mult)
        self.act = HSigmoid()

    def forward(self, x_low, x_global):
        xl_hw = paddle.shape(x_low)[2:]
        local_feat = self.local_embedding(x_low)

        global_feat = self.global_embedding(x_global)
        sig_act = F.interpolate(
            self.act(global_feat), xl_hw, mode='bilinear', align_corners=False)

        global_feat = F.interpolate(
            global_feat, xl_hw, mode='bilinear', align_corners=False)

        out = local_feat * sig_act + global_feat
        return out


class InjectionMultiSumReverse(nn.Layer):
    def __init__(self,
                 in_channels=(64, 128, 256, 384),
                 out_channels=256,
                 activations=nn.ReLU6,
                 lr_mult=1.0):
        super(InjectionMultiSumReverse, self).__init__()
        self.embedding_list = nn.LayerList()
        self.act_embedding_list = nn.LayerList()
        self.act_list = nn.LayerList()
        for i in range(len(in_channels)):
            self.embedding_list.append(
                ConvBNAct(
                    in_channels[i],
                    out_channels,
                    kernel_size=1,
                    lr_mult=lr_mult))
            self.act_embedding_list.append(
                ConvBNAct(
                    in_channels[i],
                    out_channels,
                    kernel_size=1,
                    lr_mult=lr_mult))
            if i < len(in_channels) - 1:
                self.act_list.append(HSigmoid())

    def forward(self, inputs):  # x_x8, x_x16, x_x32, x_x64
        out = []
        high_feat = self.embedding_list[0](inputs[0])
        for i in range(len(inputs) - 1):
            add_low_feat = high_feat  # x8 *256
            act = self.act_list[i](self.act_embedding_list[i](inputs[i]))
            high_feat = self.embedding_list[i + 1](inputs[i + 1])
            high_feat_up = F.interpolate(
                high_feat,
                size=add_low_feat.shape[2:],
                mode="bilinear",
                align_corners=True)

            out.append(act * high_feat_up + add_low_feat)

        return out


class InjectionLayerFusionSimple(nn.Layer):
    def __init__(self,
                 in_channels=(384, 256, 128, 64),
                 out_channels=256,
                 activations=None,
                 global_in_channels=None,
                 lr_mult=1.0):
        super(InjectionLayerFusionSimple, self).__init__()

        self.embedding_list = nn.LayerList()
        self.act_embedding_list = nn.LayerList()
        for i in range(len(in_channels)):
            self.embedding_list.append(
                ConvBNAct(
                    in_channels[i],
                    out_channels,
                    kernel_size=1,
                    lr_mult=lr_mult))
            if i < len(in_channels) - 1:
                self.act_embedding_list.append(HSigmoid())
        self.embedding_act = ConvBNAct(
            out_channels, out_channels, kernel_size=1, lr_mult=lr_mult)
        self.embedding_act1 = ConvBNAct(
            out_channels,
            out_channels,
            kernel_size=1,
            lr_mult=lr_mult,
            use_conv=False)
        self.embedding_act2 = ConvBNAct(
            out_channels,
            out_channels,
            kernel_size=1,
            lr_mult=lr_mult,
            use_conv=False)

    def forward(self, inputs):  # x_x64, x_x32, x_x16, x_x8
        high_feat_act = self.act_embedding_list[0](self.embedding_act(inputs[
            0]))
        low_feat = self.embedding_list[1](inputs[1])
        high_feat = self.embedding_list[0](inputs[0])

        res1 = high_feat_act * low_feat + high_feat

        high_feat = res1
        high_feat_act = self.embedding_act1(high_feat)
        low_feat = self.embedding_list[2](inputs[2])

        res2 = high_feat_act * low_feat + high_feat

        high_feat = res2
        high_feat_act = self.embedding_act2(high_feat)
        low_feat = self.embedding_list[3](inputs[3])
        res3 = high_feat_act * low_feat + high_feat

        return [res1, res2, res3]

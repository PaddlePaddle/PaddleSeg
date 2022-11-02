import torch
import torch.nn as nn
from isnet.common.module import RSU7, RSU6, RSU5, RSU4, RSU4F, _upsample_like, UpBlock, get_sigmoid, REBNCONV, myrebnconv
from isnet.common.model_enum import NetType


class GTNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(GTNet, self).__init__()

        # self.stage0 = nn.Conv2d(in_ch, 16, kernel_size=3, stride=2, padding=1)
        self.conv_in = myrebnconv(in_ch,16,3,stride=2,padding=1)

        self.stage1 = RSU7(16, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 32, 128)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(128, 32, 256)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(256, 64, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 64, 512)

        self.stage6up = UpBlock(512, 64, nb_Conv=2, nb_upsample=3)
        self.stage5up = UpBlock(512, 64, nb_Conv=2, nb_upsample=2)
        self.stage4up = UpBlock(256, 64, nb_Conv=2, nb_upsample=1)
        self.stage3up = UpBlock(128, 64, nb_Conv=2, nb_upsample=0)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        # self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        # self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        # self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        # self.side6 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)


    def forward(self, x):
        hx = x

        stage0 = self.conv_in(hx)
        stage1 = self.stage1(stage0)
        hx = self.pool12(stage1)
        stage2 = self.stage2(hx)
        hx = self.pool23(stage2)
        stage3 = self.stage3(hx)
        hx = self.pool34(stage3)
        stage4 = self.stage4(hx)
        hx = self.pool45(stage4)
        stage5 = self.stage5(hx)
        hx = self.pool56(stage5)
        stage6 = self.stage6(hx)

        feature_maps = (stage1, stage2, stage3, stage4, stage5, stage6)

        d1 = self.side1(stage1)
        d1 = _upsample_like(d1,x)

        d2 = self.side2(stage2)
        d2 = _upsample_like(d2,x)
    
        d3 = self.stage3up(stage3, stage2)
        d3 = self.side3(d3)
        d3 = _upsample_like(d3,x)

        d4 = self.stage4up(stage4, stage2)
        d4 = self.side4(d4)
        d4 = _upsample_like(d4,x)
    
        d5 = self.stage5up(stage5, stage2)
        d5 = self.side5(d5)
        d5 = _upsample_like(d5,x)

        d6 = self.stage6up(stage6, stage2)
        d6 = self.side6(d6)
        d6 = _upsample_like(d6,x)

        d1_sig = get_sigmoid(d1)
        d2_sig = get_sigmoid(d2)
        d3_sig = get_sigmoid(d3)
        d4_sig = get_sigmoid(d4)
        d5_sig = get_sigmoid(d5)
        d6_sig = get_sigmoid(d6)
        side_outputs = (d1_sig, d2_sig, d3_sig, d4_sig, d5_sig, d6_sig)

        return side_outputs, feature_maps


class DISNet(nn.Module):

    def __init__(self, in_ch=3, out_ch=1, model_type=NetType.DISNET):
        super(DISNet, self).__init__()
        self.model_type = model_type
        if self.model_type == NetType.DISNET:
            self.stage0 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=2, padding=1)
            self.stage1 = RSU7(64, 32, 64)
        else:
            self.stage1 = RSU7(in_ch, 32, 64)

        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        if self.model_type == NetType.DISNET:
            self.stage6up = UpBlock(512, 64, nb_Conv=2, nb_upsample=3)
            self.stage5up = UpBlock(512, 64, nb_Conv=2, nb_upsample=2)
            self.stage4up = UpBlock(256, 64, nb_Conv=2, nb_upsample=1)
            self.stage3up = UpBlock(128, 64, nb_Conv=2, nb_upsample=0)

            self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
            self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
            self.side3 = nn.Conv2d(64,out_ch,3,padding=1)
            self.side4 = nn.Conv2d(64,out_ch,3,padding=1)
            self.side5 = nn.Conv2d(64,out_ch,3,padding=1)
            self.side6 = nn.Conv2d(64,out_ch,3,padding=1)
        else:
            self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
            self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
            self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
            self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
            self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
            self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):

        hx = x

        if self.model_type == NetType.DISNET:
            hx = self.stage0(hx)
        # stage 1
        stage1 = self.stage1(hx)
        hx = self.pool12(stage1)

        # stage 2
        stage2 = self.stage2(hx)
        hx = self.pool23(stage2)

        # stage 3
        stage3 = self.stage3(hx)
        hx = self.pool34(stage3)

        # stage 4
        stage4 = self.stage4(hx)
        hx = self.pool45(stage4)

        # stage 5
        stage5 = self.stage5(hx)
        hx = self.pool56(stage5)

        # stage 6
        stage6 = self.stage6(hx)
        stage6up = _upsample_like(stage6, stage5)

        # -------------------- decoder --------------------
        stage5d = self.stage5d(torch.cat((stage6up, stage5), 1))
        stage5dup = _upsample_like(stage5d, stage4)

        stage4d = self.stage4d(torch.cat((stage5dup, stage4), 1))
        stage4dup = _upsample_like(stage4d, stage3)

        stage3d = self.stage3d(torch.cat((stage4dup, stage3), 1))
        stage3dup = _upsample_like(stage3d, stage2)

        stage2d = self.stage2d(torch.cat((stage3dup, stage2), 1))
        stage2dup = _upsample_like(stage2d, stage1)

        stage1d = self.stage1d(torch.cat((stage2dup, stage1), 1))

        feature_maps = (stage1d, stage2d, stage3d, stage4d, stage5d, stage6)

        # side output
        d1 = self.side1(stage1d)

        d2 = self.side2(stage2d)
        d2 = _upsample_like(d2, d1)
        
        stage3dup = self.stage3up(stage3d, stage2d)
        d3 = self.side3(stage3dup)
        d3 = _upsample_like(d3, d1)

        stage4dup = self.stage4up(stage4d, stage2d)
        d4 = self.side4(stage4dup)
        d4 = _upsample_like(d4, d1)

        stage5dup = self.stage5up(stage5d, stage2d)
        d5 = self.side5(stage5dup)
        d5 = _upsample_like(d5, d1)

        stage6up = self.stage6up(stage6, stage2d)
        d6 = self.side6(stage6up)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        side_outputs = (
            get_sigmoid(d0), get_sigmoid(d1), get_sigmoid(d2), get_sigmoid(d3), get_sigmoid(d4), get_sigmoid(d5),
            get_sigmoid(d6))

        return side_outputs, feature_maps


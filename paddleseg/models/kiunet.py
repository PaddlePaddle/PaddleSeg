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
from paddle.nn import initializer

def init_weights(init_type='kaiming'):
    if init_type == 'normal':
        return paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Normal())
    elif init_type == 'xavier':
        return paddle.framework.ParamAttr(initializer=paddle.nn.initializer.XavierNormal())
    elif init_type == 'kaiming':
        return paddle.framework.ParamAttr(initializer=paddle.nn.initializer.KaimingNormal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

@manager.MODELS.add_component
class KIunet(nn.Layer):
    
    """
    The KIunet implementation based on PaddlePaddle.

    The original article refers to
    Valanarasu J M J, Sindagi V A, Hacihaliloglu I, et al. Kiu-net: Towards accurate segmentation of biomedical images using over-complete representations
    (https://arxiv.org/pdf/2006.04878v2.pdf).

    Args:
        in_channels (int): The channel number of input image.
        num_classes (int): The unique number of target classes.
        """
    
    def __init__(self ,in_channels = 3, num_classes =3):
        super(KIunet,self).__init__()

        self.in_channels = in_channels
        self.n_class = num_classes

        self.encoder1 = nn.Conv2D(self.in_channels, 16, 3, stride=1, padding=1)  # First Layer GrayScale Image , change to input channels to 3 in case of RGB 
        self.en1_bn = nn.BatchNorm(16)
        self.encoder2=   nn.Conv2D(16, 32, 3, stride=1, padding=1)  
        self.en2_bn = nn.BatchNorm(32)
        self.encoder3=   nn.Conv2D(32, 64, 3, stride=1, padding=1)
        self.en3_bn = nn.BatchNorm(64)

        self.decoder1 =   nn.Conv2D(64, 32, 3, stride=1, padding=1)   
        self.de1_bn = nn.BatchNorm(32)
        self.decoder2 =   nn.Conv2D(32,16, 3, stride=1, padding=1)
        self.de2_bn = nn.BatchNorm(16)
        self.decoder3 =   nn.Conv2D(16, 8, 3, stride=1, padding=1)
        self.de3_bn = nn.BatchNorm(8)

        self.decoderf1 =   nn.Conv2D(64, 32, 3, stride=1, padding=1)
        self.def1_bn = nn.BatchNorm(32)
        self.decoderf2=   nn.Conv2D(32, 16, 3, stride=1, padding=1)
        self.def2_bn = nn.BatchNorm(16)
        self.decoderf3 =   nn.Conv2D(16, 8, 3, stride=1, padding=1)
        self.def3_bn = nn.BatchNorm(8)

        self.encoderf1 =   nn.Conv2D(in_channels, 16, 3, stride=1, padding=1)  # First Layer GrayScale Image , change to input channels to 3 in case of RGB 
        self.enf1_bn = nn.BatchNorm(16)
        self.encoderf2=   nn.Conv2D(16, 32, 3, stride=1, padding=1)
        self.enf2_bn = nn.BatchNorm(32)
        self.encoderf3 =   nn.Conv2D(32, 64, 3, stride=1, padding=1)
        self.enf3_bn = nn.BatchNorm(64)

        self.intere1_1 = nn.Conv2D(16,16,3, stride=1, padding=1)
        self.inte1_1bn = nn.BatchNorm(16)
        self.intere2_1 = nn.Conv2D(32,32,3, stride=1, padding=1)
        self.inte2_1bn = nn.BatchNorm(32)
        self.intere3_1 = nn.Conv2D(64,64,3, stride=1, padding=1)
        self.inte3_1bn = nn.BatchNorm(64)

        self.intere1_2 = nn.Conv2D(16,16,3, stride=1, padding=1)
        self.inte1_2bn = nn.BatchNorm(16)
        self.intere2_2 = nn.Conv2D(32,32,3, stride=1, padding=1)
        self.inte2_2bn = nn.BatchNorm(32)
        self.intere3_2 = nn.Conv2D(64,64,3, stride=1, padding=1)
        self.inte3_2bn = nn.BatchNorm(64)

        self.interd1_1 = nn.Conv2D(32,32,3, stride=1, padding=1)
        self.intd1_1bn = nn.BatchNorm(32)
        self.interd2_1 = nn.Conv2D(16,16,3, stride=1, padding=1)
        self.intd2_1bn = nn.BatchNorm(16)
        self.interd3_1 = nn.Conv2D(64,64,3, stride=1, padding=1)
        self.intd3_1bn = nn.BatchNorm(64)

        self.interd1_2 = nn.Conv2D(32,32,3, stride=1, padding=1)
        self.intd1_2bn = nn.BatchNorm(32)
        self.interd2_2 = nn.Conv2D(16,16,3, stride=1, padding=1)
        self.intd2_2bn = nn.BatchNorm(16)
        self.interd3_2 = nn.Conv2D(64,64,3, stride=1, padding=1)
        self.intd3_2bn = nn.BatchNorm(64)

        self.final = nn.Conv2D(8,self.n_class,1,stride=1,padding=0),

        #self.soft = nn.Softmax(dim =1)
        
        # initialise weights
        for m in self.sublayers ():
            if isinstance(m, nn.Conv2D):
                m.weight_attr = init_weights(init_type='kaiming')
                m.bias_attr = init_weights(init_type='kaiming')
            elif isinstance(m, nn.BatchNorm):
                m.param_attr =init_weights(init_type='kaiming')
                m.bias_attr = init_weights(init_type='kaiming') 

    def forward(self, x):
        # input: c * h * w -> 16 * h/2 * w/2
        out = F.relu(self.en1_bn(F.max_pool2d(self.encoder1(x),2,2)))  #U-Net branch
        # c * h * w -> 16 * 2h * 2w
        out1 = F.relu(self.enf1_bn(F.interpolate(self.encoderf1(x),scale_factor=(2,2),mode ='bicubic'))) #Ki-Net branch
        # 16 * h/2 * w/2
        tmp = out
        # 16 * 2h * 2w -> 16 * h/2 * w/2
        out = paddle.add(out,F.interpolate(F.relu(self.inte1_1bn(self.intere1_1(out1))),scale_factor=(0.25,0.25),mode ='bicubic')) #CRFB
        # 16 * h/2 * w/2 -> 16 * 2h * 2w
        out1 = paddle.add(out1,F.interpolate(F.relu(self.inte1_2bn(self.intere1_2(tmp))),scale_factor=(4,4),mode ='bicubic')) #CRFB
        
        # 16 * h/2 * w/2
        u1 = out  #skip conn
        # 16 * 2h * 2w
        o1 = out1  #skip conn

        # 16 * h/2 * w/2 -> 32 * h/4 * w/4
        out = F.relu(self.en2_bn(F.max_pool2d(self.encoder2(out),2,2)))
        # 16 * 2h * 2w -> 32 * 4h * 4w
        out1 = F.relu(self.enf2_bn(F.interpolate(self.encoderf2(out1),scale_factor=(2,2),mode ='bicubic')))
        #  32 * h/4 * w/4
        tmp = out
        # 32 * 4h * 4w -> 32 * h/4 *w/4
        out = paddle.add(out,F.interpolate(F.relu(self.inte2_1bn(self.intere2_1(out1))),scale_factor=(0.0625,0.0625),mode ='bicubic'))
        # 32 * h/4 * w/4 -> 32 *4h *4w
        out1 = paddle.add(out1,F.interpolate(F.relu(self.inte2_2bn(self.intere2_2(tmp))),scale_factor=(16,16),mode ='bicubic'))
        
        #  32 * h/4 *w/4
        u2 = out
        #  32 *4h *4w
        o2 = out1
        
        # 32 * h/4 *w/4 -> 64 * h/8 *w/8
        out = F.relu(self.en3_bn(F.max_pool2d(self.encoder3(out),2,2)))
        # 32 *4h *4w -> 64 * 8h *8w
        out1 = F.relu(self.enf3_bn(F.interpolate(self.encoderf3(out1),scale_factor=(2,2),mode ='bicubic')))
        #  64 * h/8 *w/8 
        tmp = out
        #  64 * 8h *8w -> 64 * h/8 * w/8
        out = paddle.add(out,F.interpolate(F.relu(self.inte3_1bn(self.intere3_1(out1))),scale_factor=(0.015625,0.015625),mode ='bicubic'))
        #  64 * h/8 *w/8 -> 64 * 8h * 8w
        out1 = paddle.add(out1,F.interpolate(F.relu(self.inte3_2bn(self.intere3_2(tmp))),scale_factor=(64,64),mode ='bicubic'))
        
        ### End of encoder block

        ### Start Decoder
        
        # 64 * h/8 * w/8 -> 32 * h/4 * w/4 
        out = F.relu(self.de1_bn(F.interpolate(self.decoder1(out),scale_factor=(2,2),mode ='bicubic')))  #U-NET
        # 64 * 8h * 8w -> 32 * 4h * 4w 
        out1 = F.relu(self.def1_bn(F.max_pool2d(self.decoderf1(out1),2,2))) #Ki-NET
        # 32 * h/4 * w/4 
        tmp = out
        # 32 * 4h * 4w  -> 32 * h/4 * w/4 
        out = paddle.add(out,F.interpolate(F.relu(self.intd1_1bn(self.interd1_1(out1))),scale_factor=(0.0625,0.0625),mode ='bicubic'))
        # 32 * h/4 * w/4  -> 32 * 4h * 4w 
        out1 = paddle.add(out1,F.interpolate(F.relu(self.intd1_2bn(self.interd1_2(tmp))),scale_factor=(16,16),mode ='bicubic'))
        
        # 32 * h/4 * w/4 
        out = paddle.add(out,u2)  #skip conn
        # 32 * 4h * 4w 
        out1 = paddle.add(out1,o2)  #skip conn

        # 32 * h/4 * w/4 -> 16 * h/2 * w/2 
        out = F.relu(self.de2_bn(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bicubic')))
        # 32 * 4h * 4w  -> 16 * 2h * 2w
        out1 = F.relu(self.def2_bn(F.max_pool2d(self.decoderf2(out1),2,2)))
        # 16 * h/2 * w/2 
        tmp = out
        # 16 * 2h * 2w -> 16 * h/2 * w/2
        out = paddle.add(out,F.interpolate(F.relu(self.intd2_1bn(self.interd2_1(out1))),scale_factor=(0.25,0.25),mode ='bicubic'))
        # 16 * h/2 * w/2 -> 16 * 2h * 2w
        out1 = paddle.add(out1,F.interpolate(F.relu(self.intd2_2bn(self.interd2_2(tmp))),scale_factor=(4,4),mode ='bicubic'))
        
        # 16 * h/2 * w/2
        out = paddle.add(out,u1)
        # 16 * 2h * 2w
        out1 = paddle.add(out1,o1)

        # 16 * h/2 * w/2 -> 8 * h * w
        out = F.relu(self.de3_bn(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bicubic')))
        # 16 * 2h * 2w -> 8 * h * w
        out1 = F.relu(self.def3_bn(F.max_pool2d(self.decoderf3(out1),2,2)))

        # 8 * h * w
        out = paddle.add(out,out1) # fusion of both branches

        # 最后一层
        out = F.relu(self.final(out))  #1*1 conv
        
        # out = self.soft(out)
        
        return out
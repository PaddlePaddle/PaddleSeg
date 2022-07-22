import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg import utils
from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.models.layers.layer_libs import SyncBatchNorm
from paddleseg.cvlibs.param_init import kaiming_normal_init,constant_init


class UNetConvBlock(nn.Layer):
    def __init__(self,in_channel,out_channel,padding,batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2D(in_channel, out_channel, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(SyncBatchNorm(out_channel))

        block.append(nn.Conv2D(out_channel, out_channel, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.SyncBatchNorm(out_channel))

        self.block = nn.Sequential(*block)
    def forward(self, x):
        #根据论文中的unet网络，我们知道unet每一层有都是由两个3*3的conv组成，所以我们这里把每一层的conv操作做成一个基本块
        out = self.block(x)
        return out

class UNetUpBlock(nn.Layer):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.Conv2DTranspose(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2D(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)
    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.shape
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = paddle.concat([up, crop1], 1)
        out = self.conv_block(out)

        return out

@manager.MODELS.add_component
class UnetCathy(nn.Layer):
    def __init__(self,in_chans=1,num_classes=2,depth=5,wf=6,padding=False,batch_norm=True,up_mode='upconv',pretrained=None):
        super(UnetCathy,self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_chans
        self.down_path = nn.LayerList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels=2 ** (wf + i)
        self.up_path = nn.LayerList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2D(prev_channels, num_classes, kernel_size=1)
        for sublayer in self.sublayers():
            if isinstance(sublayer, nn.Conv2D):
                kaiming_normal_init(sublayer.weight)
            elif isinstance(sublayer, (nn.BatchNorm, nn.SyncBatchNorm)):
                constant_init(sublayer.weight, value=1.0)
                constant_init(sublayer.bias, value=0.0)
    def forward(self,x):
        logit_list = []
        blocks=[]
        for i,down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
        logit=self.last(x)
        logit_list.append(logit)
        return logit_list






if __name__=="__main__":

    input=paddle.rand([1, 1, 572, 572])
    net=UnetCathy(wf=4)
    # net.apply()
    # print(net)
    output=net(input)
    print(output.shape)

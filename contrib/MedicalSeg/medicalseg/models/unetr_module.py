import paddle
import paddle.nn as nn

# yellow block in Fig.1
class Conv3DBlock(nn.Layer):
    def __init__(self, in_planes, out_planes, kernel_size=3, double=True, norm=nn.BatchNorm3D, skip=True):
        super(Conv3DBlock, self).__init__()
       
        self.skip = skip
        self.downsample = in_planes != out_planes
        self.final_activation = nn.LeakyReLU(negative_slope=0.01) 
        padding = (kernel_size - 1) // 2
        if double:
            self.conv_block = nn.Sequential(
                nn.Conv3D(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                          padding=padding),
                norm(out_planes),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv3D(out_planes, out_planes, kernel_size=kernel_size, stride=1,
                          padding=padding),
                norm(out_planes))
        else:
            self.conv_block = nn.Sequential(
                nn.Conv3D(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                          padding=padding),
                norm(out_planes))

        if self.skip and self.downsample:
            self.conv_down = nn.Sequential(
                nn.Conv3D(in_planes, out_planes, kernel_size=1, stride=1,
                          padding=0),
                norm(out_planes))

    def forward(self, x):
        y = self.conv_block(x)
        if self.skip:
            res = x
            if self.downsample:
                res = self.conv_down(res)
            y = y + res
        return self.final_activation(y)


# green block in Fig.1
class TranspConv3DBlock(nn.Layer):
    def __init__(self, in_planes, out_planes):
        super(TranspConv3DBlock, self).__init__()
        self.block = nn.Conv3DTranspose(in_planes, out_planes, kernel_size=2, stride=2,
                                        padding=0, output_padding=0,bias_attr=False)

    def forward(self, x):
        y = self.block(x)
        return y


class BlueBlock(nn.Layer):
    def __init__(self, in_planes, out_planes, layers=1, conv_block=False):
        """
        blue box in Fig.1
        Args:
            in_planes: in channels of transpose convolution
            out_planes: out channels of transpose convolution
            layers: number of blue blocks, transpose convs
            conv_block: whether to include a conv block after each transpose conv. deafaults to False
        """
        super(BlueBlock, self).__init__()
        self.blocks = nn.LayerList([TranspConv3DBlock(in_planes, out_planes),
                                            ])
        if conv_block:
            self.blocks.append(Conv3DBlock(out_planes, out_planes, double=False))

        if int(layers)>=2:
            for _ in range(int(layers) - 1):
                self.blocks.append(TranspConv3DBlock(out_planes, out_planes))
                if conv_block:
                    self.blocks.append(Conv3DBlock(out_planes, out_planes, double=False))

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
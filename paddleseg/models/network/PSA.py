import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .para_init import kaiming_normal_,kaiming_uniform_,constant_
def constant_init(module, val, bias=0):
    #hasattr函数，判断对象是否包含对应的属性
    if hasattr(module, 'weight') and module.weight is not None:
        constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        constant_(module.bias, bias)

class PSA_s(nn.Layer):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA_s, self).__init__()

        self.inplanes = inplanes
        
        self.inter_planes = planes // 2
        self.planes = planes
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size-1)//2

        self.conv_q_right = nn.Conv2D(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias_attr=False)
        self.conv_v_right = nn.Conv2D(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias_attr=False)
        self.conv_up = nn.Conv2D(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias_attr=False)
        #Softmax 参数 dim 代表需要归一化的维度
        self.softmax_right = nn.Softmax(axis=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2D(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias_attr=False)   #g
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        
        self.conv_v_left = nn.Conv2D(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias_attr=False)   #theta
        self.softmax_left = nn.Softmax(axis=2)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        #batch, channel, height, width = input_x.size()
        batch, channel, height, width = input_x.shape
        # [N, IC, H*W]  %reshape
        #input_x = input_x.view(batch, channel, height * width)
        input_x = input_x.reshape((batch, channel, height * width))

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W] 
        #context_mask = context_mask.view(batch, 1, height * width)
        context_mask = context_mask.reshape((batch, 1, height * width))

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        #context = paddle.matmul(input_x, context_mask.transpose(1,2))
        context = paddle.matmul(input_x, context_mask.transpose((0,2,1)))

        # [N, IC, 1, 1] % unsequeeze
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)
        # 圈中带点的符号代表着逐项相乘
        out = x * mask_ch

        return out

    def channel_pool(self, x):
        # [N, IC, H, W]
        g_x = self.conv_q_left(x)

        #batch, channel, height, width = g_x.size()
        batch, channel, height, width = g_x.shape

        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)

        #batch, channel, avg_x_h, avg_x_w = avg_x.size()
        batch, channel, avg_x_h, avg_x_w = avg_x.shape
        # [N, 1, IC]
        #avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        avg_x = avg_x.reshape((batch, channel, avg_x_h * avg_x_w))
        avg_x = paddle.reshape(avg_x,[batch,avg_x_h * avg_x_w,channel])
    

        # [N, IC, H*W]
        #theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)
        theta_x = self.conv_v_left(x).reshape((batch, self.inter_planes, height * width))

        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = paddle.matmul(avg_x, theta_x)
        # [N, 1, H*W]
        context = self.softmax_left(context)

        # [N, 1, H, W]
        #context = context.view(batch, 1, height, width)
        context = context.reshape((batch, 1, height, width))

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out

    def forward(self, x):
        # [N, C, H, W]
        context_channel = self.spatial_pool(x)
        # [N, C, H, W]
        context_spatial = self.channel_pool(x)
        # [N, C, H, W]
        out = context_spatial + context_channel
        return out

class PSA_p(nn.Layer):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA_p, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        ratio = 4

        self.conv_q_right = nn.Conv2D(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias_attr=False)
        self.conv_v_right = nn.Conv2D(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                      bias_attr=False)
        # self.conv_up = nn.Conv2D(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias_attr=False)
        self.conv_up = nn.Sequential(
            nn.Conv2D(self.inter_planes, self.inter_planes // ratio, kernel_size=1),
            nn.LayerNorm([self.inter_planes // ratio, 1, 1]),
            nn.ReLU(),
            nn.Conv2D(self.inter_planes // ratio, self.planes, kernel_size=1)
        )
        self.softmax_right = nn.Softmax(axis=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2D(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias_attr=False)  # g
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv_v_left = nn.Conv2D(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias_attr=False)  # theta
        self.softmax_left = nn.Softmax(axis=2)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        #batch, channel, height, width = input_x.size()
        batch, channel, height, width = input_x.shape

        # [N, IC, H*W]
        #input_x = input_x.view(batch, channel, height * width)
        input_x = input_x.reshape((batch, channel, height * width))
        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W]
        #context_mask = context_mask.view(batch, 1, height * width)
        context_mask = context_mask.reshape((batch, 1, height * width))
        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        #context = paddle.matmul(input_x, context_mask.transpose((1, 2)))
        context = paddle.matmul(input_x, context_mask.transpose((0,2,1)))
        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)
        
        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out

    def channel_pool(self, x):
        # [N, IC, H, W]
        g_x = self.conv_q_left(x)

        #batch, channel, height, width = g_x.size()
        batch, channel, height, width = g_x.shape
        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)

        #batch, channel, avg_x_h, avg_x_w = avg_x.size()
        batch, channel, avg_x_h, avg_x_w = avg_x.shape
        # [N, 1, IC]
        #avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)
        avg_x = avg_x.reshape((batch, channel, avg_x_h * avg_x_w))
        avg_x = paddle.reshape(avg_x,[batch,avg_x_h * avg_x_w,channel])

        # [N, IC, H*W]
        #theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)
        theta_x = self.conv_v_left(x).reshape((batch, self.inter_planes, height * width))

        # [N, IC, H*W]
        theta_x = self.softmax_left(theta_x)

        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = paddle.matmul(avg_x, theta_x)

        # [N, 1, H, W]
        #context = context.view(batch, 1, height, width)
        context = context.reshape((batch, 1, height, width))
        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out

    def forward(self, x):
        # [N, C, H, W]
        out = self.spatial_pool(x)

        # [N, C, H, W]
        out = self.channel_pool(out)

        # [N, C, H, W]
        # out = context_spatial + context_channel

        return out


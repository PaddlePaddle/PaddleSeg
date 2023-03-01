import paddle
import paddle.nn as nn
from paddle import Tensor
import paddle.nn.functional as F

from paddleseg.utils.weight_init import group_weight
from itertools import chain

bn_momentum = 0.9


class EncoderNetwork(nn.Layer):
    def __init__(self, back_bone, num_classes, norm_layer=nn.BatchNorm2D, pretrained_model=None):
        super(EncoderNetwork, self).__init__()

        self.backbone = back_bone

        self.head = Head(num_classes, norm_layer, bn_momentum)
        self.business_layer = []
        self.business_layer.append(self.head)

    def forward(self, data):
        blocks = self.backbone(data)

        f = self.head(blocks)
        return f


class upsample(nn.Layer):
    def __init__(self, in_channels, out_channels, data_shape,
                 norm_act=nn.BatchNorm2D):
        super(upsample, self).__init__()
        self.data_shape = data_shape
        self.classifier = nn.Conv2D(in_channels, out_channels, kernel_size=1, stride=1, bias_attr=True)
        self.last_conv = nn.Sequential(nn.Conv2D(304, 256, kernel_size=3, stride=1, padding=1, bias_attr=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1, bias_attr=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU())

    def forward(self, x, data_shape=None):
        f = self.last_conv(x)
        pred = self.classifier(f)
        if self.training:
            h, w = self.data_shape[0], self.data_shape[1]
        else:
            if data_shape is not None:
                h, w = data_shape[0], data_shape[1]
            else:
                h, w = self.data_shape[0], self.data_shape[1]

        return F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)


def _l2_normalize(d):
    d1 = paddle.flatten(d, start_axis=1)
    d_reshaped = paddle.reshape(d1, (d.shape[0], -1, *(1 for _ in range(d.ndim - 2))))
    d /= paddle.norm(d_reshaped, axis=1, keepdim=True) + 1e-8
    return d


def get_r_adv_t(x, decoder1, decoder2, it=1, xi=1e-1, eps=10.0):
    # stop bn
    decoder1.eval()
    decoder2.eval()

    x_detached = x.detach()
    # 一般用于神经网络的推理阶段, 表示张量的计算过程中无需计算梯度
    with paddle.no_grad():
        # get the ensemble results from teacher
        pred = F.softmax((decoder1(x_detached) + decoder2(x_detached)) / 2, axis=1)

    d = paddle.subtract(paddle.rand(x.shape), paddle.to_tensor(0.5))
    d = paddle.to_tensor(d, place=x.place)
    d = _l2_normalize(d)

    # assist students to find the effective va-noise
    for _ in range(it):
        # 进行更新参数
        d.stop_gradient = False
        pred_hat = (decoder1(x_detached + xi * d) + decoder2(x_detached + xi * d)) / 2
        logp_hat = F.log_softmax(pred_hat, axis=1)
        # 计算散度，KL散度可以用来衡量两个概率分布之间的相似性
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')

        adv_distance.backward()
        d = _l2_normalize(d.grad)

        params_list_1 = []
        for module in chain(decoder1.business_layer):
            params_list_1 = group_weight(params_list_1, module, nn.BatchNorm2D, lr=0, clear_flag=True)

        params_list_2 = []
        for module in chain(decoder2.business_layer):
            params_list_2 = group_weight(params_list_2, module, nn.BatchNorm2D, lr=0, clear_flag=True)

        optimizer1 = paddle.optimizer.Momentum(
            parameters=params_list_1)
        optimizer2 = paddle.optimizer.Momentum(
            parameters=params_list_2)
        optimizer1.clear_grad()
        optimizer2.clear_grad()

    r_adv = d * eps

    # reopen bn, but freeze other params.
    decoder1.train()
    decoder2.train()
    return r_adv


class DecoderNetwork(nn.Layer):
    def __init__(self, num_classes,
                 data_shape,
                 conv_in_ch=256):
        super(DecoderNetwork, self).__init__()
        self.upsample = upsample(conv_in_ch, num_classes, norm_act=nn.BatchNorm2D,
                                 data_shape=data_shape)
        self.business_layer = []
        self.business_layer.append(self.upsample.last_conv)
        self.business_layer.append(self.upsample.classifier)

    def forward(self, f, data_shape=None):
        pred = self.upsample(f, data_shape)
        return pred


class VATDecoderNetwork(nn.Layer):
    def __init__(self, num_classes,
                 data_shape,
                 conv_in_ch=256):
        super(VATDecoderNetwork, self).__init__()
        self.upsample = upsample(conv_in_ch, num_classes, norm_act=nn.BatchNorm2D,
                                 data_shape=data_shape)
        self.business_layer = []
        self.business_layer.append(self.upsample.last_conv)
        self.business_layer.append(self.upsample.classifier)

    def forward(self, f, data_shape=None, t_model=None):
        if t_model is not None:
            r_adv = get_r_adv_t(f, t_model[0], t_model[1], it=1, xi=1e-6, eps=2.0)
            f = f + r_adv

        pred = self.upsample(f, data_shape=data_shape)
        return pred


#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
class ASPP(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation_rates=(12, 24, 36),
                 hidden_channels=256,
                 norm_act=nn.BatchNorm2D,
                 pooling_size=None):
        super(ASPP, self).__init__()
        self.pooling_size = pooling_size

        self.map_convs = nn.LayerList([
            nn.Conv2D(in_channels, hidden_channels, 1, bias_attr=False),
            nn.Conv2D(in_channels, hidden_channels, 3, bias_attr=False, dilation=dilation_rates[0],
                      padding=dilation_rates[0]),
            nn.Conv2D(in_channels, hidden_channels, 3, bias_attr=False, dilation=dilation_rates[1],
                      padding=dilation_rates[1]),
            nn.Conv2D(in_channels, hidden_channels, 3, bias_attr=False, dilation=dilation_rates[2],
                      padding=dilation_rates[2]),
        ])
        self.map_bn = norm_act(hidden_channels * 4)

        self.global_pooling_conv = nn.Conv2D(in_channels, hidden_channels, 1, bias_attr=False)
        self.global_pooling_bn = norm_act(hidden_channels)

        self.red_conv = nn.Conv2D(hidden_channels * 4, out_channels, 1, bias_attr=False)
        self.pool_red_conv = nn.Conv2D(hidden_channels, out_channels, 1, bias_attr=False)
        self.red_bn = norm_act(out_channels)

        self.leak_relu = nn.LeakyReLU()

    def forward(self, x: Tensor) -> Tensor:
        # Map convolutions
        out = paddle.concat([m(x) for m in self.map_convs], axis=1)
        out = self.map_bn(out)
        out = self.leak_relu(out)  # add activation layer
        out = self.red_conv(out)

        # Global pooling
        pool = self._global_pooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)
        pool = self.leak_relu(pool)  # add activation layer
        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            repeats = paddle.to_tensor([x.shape[2]], dtype='int32')
            pool = paddle.repeat_interleave(pool, repeats, axis=2)
            repeats = paddle.to_tensor([x.shape[3]], dtype='int32')
            pool = paddle.repeat_interleave(pool, repeats, axis=3)

        out = out + pool
        out = self.red_bn(out)
        out = self.leak_relu(out)  # add activation layer
        return out

    def _global_pooling(self, x: Tensor) -> Tensor:
        if self.training or self.pooling_size is None:
            pool = paddle.flatten(x, start_axis=-2).mean(axis=-1)
            pool = paddle.reshape(pool, (pool.shape[0], pool.shape[1], 1, 1))

        else:
            raise NotImplementedError
        return pool


class Head(nn.Layer):
    def __init__(self, classify_classes, norm_act=nn.BatchNorm2D, bn_momentum=0.9997):
        super(Head, self).__init__()
        self.classify_classes = classify_classes
        self.aspp = ASPP(2048, 256, [6, 12, 18], norm_act=norm_act)
        self.reduce = nn.Sequential(
            nn.Conv2D(256, 48, 1, bias_attr=False),
            norm_act(48, momentum=bn_momentum),
            nn.ReLU(),
        )
        self.last_conv = nn.Sequential(nn.Conv2D(304, 256, kernel_size=3, stride=1, padding=1, bias_attr=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1, bias_attr=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       )

    def forward(self, f_list):
        f = f_list[-1]
        f = self.aspp(f)
        low_level_features = f_list[0]
        low_h, low_w = low_level_features.shape[2], low_level_features.shape[3]
        low_level_features = self.reduce(low_level_features)
        f = F.interpolate(f, size=(low_h, low_w),
                          mode='bilinear', align_corners=True)
        f = paddle.concat([f, low_level_features], axis=1)

        return f

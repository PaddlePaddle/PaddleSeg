import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class SpatialGather_Module(nn.Layer):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        b, c, h, w = probs.shape
        probs = probs.reshape([b, c, -1])
        feats = feats.reshape([b, feats.shape[1], -1])
        feats = feats.transpose([0, 2, 1])
        probs = F.softmax(self.scale * probs, axis=2)
        ocr_context = paddle.unsqueeze(paddle.matmul(probs, feats).transpose([0, 2, 1]), 3)
        return ocr_context


class SpatialOCR_Module(nn.Layer):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 norm_layer=nn.BatchNorm2D,
                 align_corners=True,
                 ):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, key_channels, scale,
                                                           norm_layer, align_corners)
        _in_channels = 2 * in_channels
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2D(_in_channels, out_channels, kernel_size=1, padding=0, bias_attr=False),
            nn.Sequential(norm_layer(out_channels), nn.ReLU()),
            nn.Dropout2D(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(paddle.concat([context, feats], 1))
        return output


class ObjectAttentionBlock2D(nn.Layer):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''

    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 norm_layer=nn.BatchNorm2D,
                 align_corners=True):
        super(ObjectAttentionBlock2D, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.align_corners = align_corners

        self.pool = nn.MaxPool2D(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2D(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1,padding=0,bias_attr=False),
            nn.Sequential(norm_layer(self.key_channels), nn.ReLU()),
            nn.Conv2D(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias_attr=False),
            nn.Sequential(norm_layer(self.key_channels), nn.ReLU())
        )
        self.f_object = nn.Sequential(
            nn.Conv2D(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias_attr=False),
            nn.Sequential(norm_layer(self.key_channels),  nn.ReLU()),
            nn.Conv2D(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias_attr=False),
            nn.Sequential(norm_layer(self.key_channels), nn.ReLU())
        )
        self.f_down = nn.Sequential(
            nn.Conv2D(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias_attr=False),
            nn.Sequential(norm_layer(self.key_channels), nn.ReLU())
        )
        self.f_up = nn.Sequential(
            nn.Conv2D(in_channels=self.key_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0, bias_attr=False),
            nn.Sequential(norm_layer(self.in_channels), nn.ReLU())
        )

    def forward(self, x, proxy):
        b, c, h, w = x.shape
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).reshape([b, self.key_channels, -1])
        query = query.transpose([0, 2, 1])
        key = self.f_object(proxy).reshape([b, self.key_channels, -1])
        value = self.f_down(proxy).reshape([b, self.key_channels, -1])
        value = value.transpose([0, 2, 1])

        sim_map = paddle.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, axis=-1)

        context = paddle.matmul(sim_map, value)
        context = context.transpose([0, 2, 1])
        context = context.reshape([b, self.key_channels, h, w])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(context, size=[h, w], mode='bilinear', align_corners=self.align_corners)
        return context


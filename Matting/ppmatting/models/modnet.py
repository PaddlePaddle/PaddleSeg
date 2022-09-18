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

from collections import defaultdict

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import scipy
import paddleseg
from paddleseg.models import layers, losses
from paddleseg import utils
from paddleseg.cvlibs import manager, param_init


@manager.MODELS.add_component
class MODNet(nn.Layer):
    """
    The MODNet implementation based on PaddlePaddle.

    The original article refers to
    Zhanghan Ke, et, al. "Is a Green Screen Really Necessary for Real-Time Portrait Matting?"
    (https://arxiv.org/pdf/2011.11961.pdf).

    Args:
        backbone: backbone model.
        hr(int, optional): The channels of high resolutions branch. Defautl: None.
        pretrained(str, optional): The path of pretrianed model. Defautl: None.

    """

    def __init__(self, backbone, hr_channels=32, pretrained=None):
        super().__init__()
        self.backbone = backbone
        self.pretrained = pretrained
        self.head = MODNetHead(
            hr_channels=hr_channels, backbone_channels=backbone.feat_channels)
        self.init_weight()
        self.blurer = GaussianBlurLayer(1, 3)
        self.loss_func_dict = None

    def forward(self, inputs):
        """
        If training, return a dict.
        If evaluation, return the final alpha prediction.
        """
        x = inputs['img']
        feat_list = self.backbone(x)
        y = self.head(inputs=inputs, feat_list=feat_list)
        if self.training:
            loss = self.loss(y, inputs)
            return y, loss
        else:
            return y

    def loss(self, logit_dict, label_dict, loss_func_dict=None):
        if loss_func_dict is None:
            if self.loss_func_dict is None:
                self.loss_func_dict = defaultdict(list)
                self.loss_func_dict['semantic'].append(paddleseg.models.MSELoss(
                ))
                self.loss_func_dict['detail'].append(paddleseg.models.L1Loss())
                self.loss_func_dict['fusion'].append(paddleseg.models.L1Loss())
                self.loss_func_dict['fusion'].append(paddleseg.models.L1Loss())
        else:
            self.loss_func_dict = loss_func_dict

        loss = {}
        # semantic loss
        semantic_gt = F.interpolate(
            label_dict['alpha'],
            scale_factor=1 / 16,
            mode='bilinear',
            align_corners=False)
        semantic_gt = self.blurer(semantic_gt)
        #         semantic_gt.stop_gradient=True
        loss['semantic'] = self.loss_func_dict['semantic'][0](
            logit_dict['semantic'], semantic_gt)

        # detail loss
        trimap = label_dict['trimap']
        mask = (trimap == 128).astype('float32')
        logit_detail = logit_dict['detail'] * mask
        label_detail = label_dict['alpha'] * mask
        loss_detail = self.loss_func_dict['detail'][0](logit_detail,
                                                       label_detail)
        loss_detail = loss_detail / (mask.mean() + 1e-6)
        loss['detail'] = 10 * loss_detail

        # fusion loss
        matte = logit_dict['matte']
        alpha = label_dict['alpha']
        transition_mask = label_dict['trimap'] == 128
        matte_boundary = paddle.where(transition_mask, matte, alpha)
        # l1 loss
        loss_fusion_l1 = self.loss_func_dict['fusion'][0](
            matte, alpha) + 4 * self.loss_func_dict['fusion'][0](matte_boundary,
                                                                 alpha)
        # composition loss
        loss_fusion_comp = self.loss_func_dict['fusion'][1](
            matte * label_dict['img'], alpha *
            label_dict['img']) + 4 * self.loss_func_dict['fusion'][1](
                matte_boundary * label_dict['img'], alpha * label_dict['img'])
        # consisten loss with semantic
        transition_mask = F.interpolate(
            label_dict['trimap'],
            scale_factor=1 / 16,
            mode='nearest',
            align_corners=False)
        transition_mask = transition_mask == 128
        matte_con_sem = F.interpolate(
            matte, scale_factor=1 / 16, mode='bilinear', align_corners=False)
        matte_con_sem = self.blurer(matte_con_sem)
        logit_semantic = logit_dict['semantic'].clone()
        logit_semantic.stop_gradient = True
        matte_con_sem = paddle.where(transition_mask, logit_semantic,
                                     matte_con_sem)
        if False:
            import cv2
            matte_con_sem_num = matte_con_sem.numpy()
            matte_con_sem_num = matte_con_sem_num[0].squeeze()
            matte_con_sem_num = (matte_con_sem_num * 255).astype('uint8')
            semantic = logit_dict['semantic'].numpy()
            semantic = semantic[0].squeeze()
            semantic = (semantic * 255).astype('uint8')
            transition_mask = transition_mask.astype('uint8')
            transition_mask = transition_mask.numpy()
            transition_mask = (transition_mask[0].squeeze()) * 255
            cv2.imwrite('matte_con.png', matte_con_sem_num)
            cv2.imwrite('semantic.png', semantic)
            cv2.imwrite('transition.png', transition_mask)
        mse_loss = paddleseg.models.MSELoss()
        loss_fusion_con_sem = mse_loss(matte_con_sem, logit_dict['semantic'])
        loss_fusion = loss_fusion_l1 + loss_fusion_comp + loss_fusion_con_sem
        loss['fusion'] = loss_fusion
        loss['fusion_l1'] = loss_fusion_l1
        loss['fusion_comp'] = loss_fusion_comp
        loss['fusion_con_sem'] = loss_fusion_con_sem

        loss['all'] = loss['semantic'] + loss['detail'] + loss['fusion']

        return loss

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class MODNetHead(nn.Layer):
    def __init__(self, hr_channels, backbone_channels):
        super().__init__()

        self.lr_branch = LRBranch(backbone_channels)
        self.hr_branch = HRBranch(hr_channels, backbone_channels)
        self.f_branch = FusionBranch(hr_channels, backbone_channels)
        self.init_weight()

    def forward(self, inputs, feat_list):
        pred_semantic, lr8x, [enc2x, enc4x] = self.lr_branch(feat_list)
        pred_detail, hr2x = self.hr_branch(inputs['img'], enc2x, enc4x, lr8x)
        pred_matte = self.f_branch(inputs['img'], lr8x, hr2x)

        if self.training:
            logit_dict = {
                'semantic': pred_semantic,
                'detail': pred_detail,
                'matte': pred_matte
            }
            return logit_dict
        else:
            return pred_matte

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.kaiming_uniform(layer.weight)


class FusionBranch(nn.Layer):
    def __init__(self, hr_channels, enc_channels):
        super().__init__()
        self.conv_lr4x = Conv2dIBNormRelu(
            enc_channels[2], hr_channels, 5, stride=1, padding=2)

        self.conv_f2x = Conv2dIBNormRelu(
            2 * hr_channels, hr_channels, 3, stride=1, padding=1)
        self.conv_f = nn.Sequential(
            Conv2dIBNormRelu(
                hr_channels + 3, int(hr_channels / 2), 3, stride=1, padding=1),
            Conv2dIBNormRelu(
                int(hr_channels / 2),
                1,
                1,
                stride=1,
                padding=0,
                with_ibn=False,
                with_relu=False))

    def forward(self, img, lr8x, hr2x):
        lr4x = F.interpolate(
            lr8x, scale_factor=2, mode='bilinear', align_corners=False)
        lr4x = self.conv_lr4x(lr4x)
        lr2x = F.interpolate(
            lr4x, scale_factor=2, mode='bilinear', align_corners=False)

        f2x = self.conv_f2x(paddle.concat((lr2x, hr2x), axis=1))
        f = F.interpolate(
            f2x, scale_factor=2, mode='bilinear', align_corners=False)
        f = self.conv_f(paddle.concat((f, img), axis=1))
        pred_matte = F.sigmoid(f)

        return pred_matte


class HRBranch(nn.Layer):
    """
    High Resolution Branch of MODNet
    """

    def __init__(self, hr_channels, enc_channels):
        super().__init__()

        self.tohr_enc2x = Conv2dIBNormRelu(
            enc_channels[0], hr_channels, 1, stride=1, padding=0)
        self.conv_enc2x = Conv2dIBNormRelu(
            hr_channels + 3, hr_channels, 3, stride=2, padding=1)

        self.tohr_enc4x = Conv2dIBNormRelu(
            enc_channels[1], hr_channels, 1, stride=1, padding=0)
        self.conv_enc4x = Conv2dIBNormRelu(
            2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1)

        self.conv_hr4x = nn.Sequential(
            Conv2dIBNormRelu(
                2 * hr_channels + enc_channels[2] + 3,
                2 * hr_channels,
                3,
                stride=1,
                padding=1),
            Conv2dIBNormRelu(
                2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(
                2 * hr_channels, hr_channels, 3, stride=1, padding=1))

        self.conv_hr2x = nn.Sequential(
            Conv2dIBNormRelu(
                2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(
                2 * hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(
                hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(
                hr_channels, hr_channels, 3, stride=1, padding=1))

        self.conv_hr = nn.Sequential(
            Conv2dIBNormRelu(
                hr_channels + 3, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(
                hr_channels,
                1,
                1,
                stride=1,
                padding=0,
                with_ibn=False,
                with_relu=False))

    def forward(self, img, enc2x, enc4x, lr8x):
        img2x = F.interpolate(
            img, scale_factor=1 / 2, mode='bilinear', align_corners=False)
        img4x = F.interpolate(
            img, scale_factor=1 / 4, mode='bilinear', align_corners=False)

        enc2x = self.tohr_enc2x(enc2x)
        hr4x = self.conv_enc2x(paddle.concat((img2x, enc2x), axis=1))

        enc4x = self.tohr_enc4x(enc4x)
        hr4x = self.conv_enc4x(paddle.concat((hr4x, enc4x), axis=1))

        lr4x = F.interpolate(
            lr8x, scale_factor=2, mode='bilinear', align_corners=False)
        hr4x = self.conv_hr4x(paddle.concat((hr4x, lr4x, img4x), axis=1))

        hr2x = F.interpolate(
            hr4x, scale_factor=2, mode='bilinear', align_corners=False)
        hr2x = self.conv_hr2x(paddle.concat((hr2x, enc2x), axis=1))

        pred_detail = None
        if self.training:
            hr = F.interpolate(
                hr2x, scale_factor=2, mode='bilinear', align_corners=False)
            hr = self.conv_hr(paddle.concat((hr, img), axis=1))
            pred_detail = F.sigmoid(hr)

        return pred_detail, hr2x


class LRBranch(nn.Layer):
    def __init__(self, backbone_channels):
        super().__init__()
        self.se_block = SEBlock(backbone_channels[4], reduction=4)
        self.conv_lr16x = Conv2dIBNormRelu(
            backbone_channels[4], backbone_channels[3], 5, stride=1, padding=2)
        self.conv_lr8x = Conv2dIBNormRelu(
            backbone_channels[3], backbone_channels[2], 5, stride=1, padding=2)
        self.conv_lr = Conv2dIBNormRelu(
            backbone_channels[2],
            1,
            3,
            stride=2,
            padding=1,
            with_ibn=False,
            with_relu=False)

    def forward(self, feat_list):
        enc2x, enc4x, enc32x = feat_list[0], feat_list[1], feat_list[4]

        enc32x = self.se_block(enc32x)
        lr16x = F.interpolate(
            enc32x, scale_factor=2, mode='bilinear', align_corners=False)
        lr16x = self.conv_lr16x(lr16x)
        lr8x = F.interpolate(
            lr16x, scale_factor=2, mode='bilinear', align_corners=False)
        lr8x = self.conv_lr8x(lr8x)

        pred_semantic = None
        if self.training:
            lr = self.conv_lr(lr8x)
            pred_semantic = F.sigmoid(lr)

        return pred_semantic, lr8x, [enc2x, enc4x]


class IBNorm(nn.Layer):
    """
    Combine Instance Norm and Batch Norm into One Layer
    """

    def __init__(self, in_channels):
        super().__init__()
        self.bnorm_channels = in_channels // 2
        self.inorm_channels = in_channels - self.bnorm_channels

        self.bnorm = nn.BatchNorm2D(self.bnorm_channels)
        self.inorm = nn.InstanceNorm2D(self.inorm_channels)

    def forward(self, x):
        bn_x = self.bnorm(x[:, :self.bnorm_channels, :, :])
        in_x = self.inorm(x[:, self.bnorm_channels:, :, :])

        return paddle.concat((bn_x, in_x), 1)


class Conv2dIBNormRelu(nn.Layer):
    """
    Convolution + IBNorm + Relu
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias_attr=None,
                 with_ibn=True,
                 with_relu=True):

        super().__init__()

        layers = [
            nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias_attr=bias_attr)
        ]

        if with_ibn:
            layers.append(IBNorm(out_channels))

        if with_relu:
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SEBlock(nn.Layer):
    """
    SE Block Proposed in https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, num_channels, reduction=1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2D(1)
        self.conv = nn.Sequential(
            nn.Conv2D(
                num_channels,
                int(num_channels // reduction),
                1,
                bias_attr=False),
            nn.ReLU(),
            nn.Conv2D(
                int(num_channels // reduction),
                num_channels,
                1,
                bias_attr=False),
            nn.Sigmoid())

    def forward(self, x):
        w = self.pool(x)
        w = self.conv(w)
        return w * x


class GaussianBlurLayer(nn.Layer):
    """ Add Gaussian Blur to a 4D tensors
    This layer takes a 4D tensor of {N, C, H, W} as input.
    The Gaussian blur will be performed in given channel number (C) splitly.
    """

    def __init__(self, channels, kernel_size):
        """
        Args:
            channels (int): Channel for input tensor
            kernel_size (int): Size of the kernel used in blurring
        """

        super(GaussianBlurLayer, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 != 0

        self.op = nn.Sequential(
            nn.Pad2D(
                int(self.kernel_size / 2), mode='reflect'),
            nn.Conv2D(
                channels,
                channels,
                self.kernel_size,
                stride=1,
                padding=0,
                bias_attr=False,
                groups=channels))

        self._init_kernel()
        self.op[1].weight.stop_gradient = True

    def forward(self, x):
        """
        Args:
            x (paddle.Tensor): input 4D tensor
        Returns:
            paddle.Tensor: Blurred version of the input
        """

        if not len(list(x.shape)) == 4:
            print('\'GaussianBlurLayer\' requires a 4D tensor as input\n')
            exit()
        elif not x.shape[1] == self.channels:
            print('In \'GaussianBlurLayer\', the required channel ({0}) is'
                  'not the same as input ({1})\n'.format(self.channels, x.shape[
                      1]))
            exit()

        return self.op(x)

    def _init_kernel(self):
        sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8

        n = np.zeros((self.kernel_size, self.kernel_size))
        i = int(self.kernel_size / 2)
        n[i, i] = 1
        kernel = scipy.ndimage.gaussian_filter(n, sigma)
        kernel = kernel.astype('float32')
        kernel = kernel[np.newaxis, np.newaxis, :, :]
        paddle.assign(kernel, self.op[1].weight)

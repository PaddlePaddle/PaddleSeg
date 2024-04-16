# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddleseg import utils
from paddleseg.cvlibs import manager
from paddleseg.models import layers
from .upernet import UPerNetHead


@manager.MODELS.add_component
class KNet(nn.Layer):
    """
    The KNet implementation based on PaddlePaddle.

    The original article refers to
    Wenwei Zhang, et, al. "K-Net: Towards Unified Image Segmentation"
    (https://arxiv.org/abs/2106.14855).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (paddle.nn.Layer): Backbone network.
        backbone_indices (tuple): Four values in the tuple indicate the indices of output of backbone.
        kernel_update_head_params (dict): The params to build KernelUpdateHead.
        kernel_generate_head_params (dict): The params to build KernelGenerateHead.
        num_stages (int, optional): The num of KernelUpdateHead. Default: 3
        channels (int, optional): The channels of intermediate layers. Default: 512.
        enable_auxiliary_loss (bool, optional): A bool value that indicates whether or not to add auxiliary loss. Default: False.
        align_corners (bool, optional): An argument of "F.interpolate". It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        dropout_prob (float, optional): Dropout ratio for KNet model. Default: 0.1.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices,
                 kernel_update_head_params,
                 kernel_generate_head_params,
                 num_stages=3,
                 channels=512,
                 enable_auxiliary_loss=False,
                 align_corners=False,
                 dropout_prob=0.1,
                 pretrained=None):
        super().__init__()
        assert hasattr(backbone, 'feat_channels'), \
            "The backbone should has `feat_channels`."
        assert len(backbone.feat_channels) >= len(backbone_indices), \
            f"The length of input `backbone_indices` ({len(backbone_indices)}) should not be" \
            f"greater than the length of `feat_channels` ({len(backbone.feat_channels)})."
        assert len(backbone.feat_channels) > max(backbone_indices), \
            f"The maximum value ({max(backbone_indices)}) of `backbone_indices` should be " \
            f"less than the length of `feat_channels` ({len(backbone.feat_channels)})."

        self.backbone = backbone
        self.backbone_indices = backbone_indices
        self.in_channels = [
            self.backbone.feat_channels[i] for i in backbone_indices
        ]

        self.align_corners = align_corners
        self.pretrained = pretrained
        self.enable_auxiliary_loss = enable_auxiliary_loss
        self.num_stages = num_stages
        self.kernel_update_head = nn.LayerList([
            KernelUpdateHead(**kernel_update_head_params)
            for _ in range(num_stages)
        ])
        self.kernel_generate_head = build_kernel_generate_head(
            kernel_generate_head_params)
        if self.enable_auxiliary_loss:
            self.aux_head = layers.AuxLayer(1024,
                                            256,
                                            num_classes,
                                            dropout_prob=dropout_prob)
        self.init_weight()

    def forward(self, x):
        feats = self.backbone(x)
        if self.enable_auxiliary_loss:
            aux_out = self.aux_head(feats[2])
        feats = [feats[i] for i in self.backbone_indices]
        sem_seg, feats, seg_kernels = self.kernel_generate_head(feats)
        stage_segs = [sem_seg]
        for i in range(self.num_stages):
            sem_seg, seg_kernels = self.kernel_update_head[i](feats,
                                                              seg_kernels,
                                                              sem_seg)
            stage_segs.append(sem_seg)
        if self.training:
            if self.enable_auxiliary_loss:
                stage_segs.append(aux_out)
            for i, v in enumerate(stage_segs):
                stage_segs[i] = F.interpolate(v,
                                              x.shape[2:],
                                              mode='bilinear',
                                              align_corners=self.align_corners)
            return stage_segs
        # only return the prediction of the last stage during testing
        return [
            F.interpolate(stage_segs[-1],
                          x.shape[2:],
                          mode='bilinear',
                          align_corners=self.align_corners)
        ]

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


def build_kernel_generate_head(kwargs):
    support_heads = ['UPerKernelHead', 'FCNKernelHead']
    head_layer = kwargs.pop('head_layer')
    assert head_layer in support_heads, f"head layer {head_layer} not supported"
    if head_layer == 'UPerKernelHead':
        return UPerKernelHead(**kwargs)
    if head_layer == 'FCNKernelHead':
        return FCNKernelHead(**kwargs)


class UPerKernelHead(UPerNetHead):

    def forward(self, inputs):
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(inputs[i]))

        laterals.append(self.ppm(inputs[-1]))
        fpn_levels = len(laterals)
        for i in range(fpn_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=prev_shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        fpn_outs = []
        for i in range(fpn_levels - 1):
            fpn_outs.append(self.fpn_convs[i](laterals[i]))
        fpn_outs.append(laterals[-1])

        for i in range(fpn_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(fpn_outs[i],
                                        size=fpn_outs[0].shape[2:],
                                        mode='bilinear',
                                        align_corners=self.align_corners)
        fuse_out = paddle.concat(fpn_outs, axis=1)
        feats = self.fpn_bottleneck(fuse_out)
        output = self.conv_seg(feats)
        if self.training:
            seg_kernels = self.conv_seg.weight.clone()
        else:
            # Since tensor.clone() raises error when exporting static model, we use tensor instead,
            # although this may cause little performance drop in mIoU.
            seg_kernels = self.conv_seg.weight
        seg_kernels = seg_kernels[None].expand(
            [feats.shape[0], *seg_kernels.shape])
        return output, feats, seg_kernels


class FCNKernelHead(nn.Layer):

    def __init__(self,
                 in_channels=2048,
                 channels=512,
                 num_convs=2,
                 concat_input=True,
                 dropout_prob=0.1,
                 num_classes=150,
                 kernel_size=3,
                 dilation=1):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.in_channels = in_channels
        self.channels = channels
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FCNKernelHead, self).__init__()
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        for i in range(num_convs):
            _in_channels = self.in_channels if i == 0 else self.channels
            convs.append(
                layers.ConvBNReLU(_in_channels,
                                  self.channels,
                                  kernel_size=kernel_size,
                                  padding=conv_padding,
                                  dilation=dilation))

        if len(convs) == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = layers.ConvBNReLU(self.in_channels + self.channels,
                                              self.channels,
                                              kernel_size=kernel_size,
                                              padding=kernel_size // 2)

        self.conv_seg = nn.Conv2D(channels, num_classes, kernel_size=1)

        if dropout_prob > 0:
            self.dropout = nn.Dropout2D(dropout_prob)
        else:
            self.dropout = None

    def forward(self, inputs):
        feats = self.convs(inputs[0])
        if self.concat_input:
            feats = self.conv_cat(paddle.concat([inputs[0], feats], axis=1))
        if self.dropout is not None:
            feats = self.dropout(feats)
        output = self.conv_seg(feats)
        if self.training:
            seg_kernels = self.conv_seg.weight.clone()
        else:
            # Since tensor.clone() raises error when exporting static model, we use tensor.detach() instead,
            # although this may cause little performance drop in mIoU.
            seg_kernels = self.conv_seg.weight
        seg_kernels = seg_kernels[None].expand(
            [feats.shape[0], *seg_kernels.shape])
        return output, feats, seg_kernels


class FFN(nn.Layer):

    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 act_fn=nn.ReLU,
                 ffn_drop=0.,
                 dropout_layer=None,
                 add_identity=True):
        super().__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.activate = act_fn()

        layers = []
        in_channels = embed_dims
        layers.append(
            nn.Sequential(nn.Linear(in_channels, feedforward_channels),
                          self.activate, nn.Dropout(ffn_drop)))
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = nn.Dropout() if dropout_layer else nn.Identity()
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class KernelUpdator(nn.Layer):

    def __init__(
        self,
        in_channels=256,
        feat_channels=64,
        out_channels=None,
        input_feat_shape=3,
        gate_sigmoid=True,
        gate_norm_act=False,
        activate_out=False,
        norm_fn=nn.LayerNorm,
        act_fn=nn.ReLU,
    ):
        super(KernelUpdator, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels_raw = out_channels
        self.gate_sigmoid = gate_sigmoid
        self.gate_norm_act = gate_norm_act
        self.activate_out = activate_out
        if isinstance(input_feat_shape, int):
            input_feat_shape = [input_feat_shape] * 2
        self.input_feat_shape = input_feat_shape
        self.act_fn = act_fn
        self.norm_fn = norm_fn
        self.out_channels = out_channels if out_channels else in_channels

        self.num_params_in = self.feat_channels
        self.num_params_out = self.feat_channels
        self.dynamic_layer = nn.Linear(self.in_channels,
                                       self.num_params_in + self.num_params_out)
        self.input_layer = nn.Linear(self.in_channels,
                                     self.num_params_in + self.num_params_out)
        self.input_gate = nn.Linear(self.in_channels, self.feat_channels)
        self.update_gate = nn.Linear(self.in_channels, self.feat_channels)
        if self.gate_norm_act:
            self.gate_norm = self.norm_fn(self.feat_channels)

        self.norm_in = self.norm_fn(self.feat_channels)
        self.norm_out = self.norm_fn(self.feat_channels)
        self.input_norm_in = self.norm_fn(self.feat_channels)
        self.input_norm_out = self.norm_fn(self.feat_channels)

        self.activation = self.act_fn()

        self.fc_layer = nn.Linear(self.feat_channels, self.out_channels)
        self.fc_norm = self.norm_fn(self.out_channels)

    def forward(self, update_feature, input_feature):
        update_feature = update_feature.reshape([-1, self.in_channels])
        num_proposals = update_feature.shape[0]
        # dynamic_layer works for
        # phi_1 and psi_3 in Eq.(4) and (5) of K-Net paper
        parameters = self.dynamic_layer(update_feature)
        param_in = parameters[:, :self.num_params_in].reshape(
            [-1, self.feat_channels])
        param_out = parameters[:, -self.num_params_out:].reshape(
            [-1, self.feat_channels])

        # input_layer works for
        # phi_2 and psi_4 in Eq.(4) and (5) of K-Net paper
        input_feats = self.input_layer(
            input_feature.reshape([num_proposals, -1, self.feat_channels]))
        input_in = input_feats[..., :self.num_params_in]
        input_out = input_feats[..., -self.num_params_out:]

        # `gate_feats` is F^G in K-Net paper
        gate_feats = input_in * param_in.unsqueeze(-2)
        if self.gate_norm_act:
            gate_feats = self.activation(self.gate_norm(gate_feats))

        input_gate = self.input_norm_in(self.input_gate(gate_feats))
        update_gate = self.norm_in(self.update_gate(gate_feats))
        if self.gate_sigmoid:
            input_gate = F.sigmoid(input_gate)
            update_gate = F.sigmoid(update_gate)
        param_out = self.norm_out(param_out)
        input_out = self.input_norm_out(input_out)

        if self.activate_out:
            param_out = self.activation(param_out)
            input_out = self.activation(input_out)

        # Gate mechanism. Eq.(5) in original paper.
        # param_out has shape (batch_size, feat_channels, out_channels)
        features = update_gate * param_out.unsqueeze(
            -2) + input_gate * input_out

        features = self.fc_layer(features)
        features = self.fc_norm(features)
        features = self.activation(features)

        return features


class KernelUpdateHead(nn.Layer):

    def __init__(self,
                 num_classes=150,
                 num_ffn_fcs=2,
                 num_heads=8,
                 num_mask_fcs=3,
                 feedforward_channels=2048,
                 in_channels=256,
                 out_channels=256,
                 dropout=0.0,
                 act_fn=nn.ReLU,
                 ffn_act_fn=nn.ReLU,
                 conv_kernel_size=3,
                 feat_transform=False,
                 kernel_init=False,
                 with_ffn=True,
                 feat_gather_stride=1,
                 mask_transform_stride=1,
                 kernel_updator_cfg=None):
        super(KernelUpdateHead, self).__init__()
        if kernel_updator_cfg is None:
            kernel_updator_cfg = dict(in_channels=256,
                                      feat_channels=256,
                                      out_channels=256,
                                      input_feat_shape=3)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.num_heads = num_heads
        self.kernel_init = kernel_init
        self.with_ffn = with_ffn
        self.conv_kernel_size = conv_kernel_size
        self.feat_gather_stride = feat_gather_stride
        self.mask_transform_stride = mask_transform_stride

        self.attention = nn.MultiHeadAttention(in_channels *
                                               conv_kernel_size**2,
                                               num_heads,
                                               dropout,
                                               bias_attr=True)
        self.attention_norm = nn.LayerNorm(in_channels * conv_kernel_size**2)

        self.kernel_update_conv = KernelUpdator(**kernel_updator_cfg)

        if feat_transform is not None:
            kernel_size = 1
            transform_channels = in_channels
            self.feat_transform = nn.Conv2D(transform_channels,
                                            in_channels,
                                            kernel_size,
                                            stride=feat_gather_stride,
                                            padding=int(feat_gather_stride //
                                                        2))
        else:
            self.feat_transform = None

        if self.with_ffn:
            self.ffn = FFN(in_channels,
                           feedforward_channels,
                           act_fn=ffn_act_fn,
                           ffn_drop=dropout)
            self.ffn_norm = nn.LayerNorm(in_channels)

        self.mask_fcs = nn.LayerList()
        for _ in range(num_mask_fcs):
            self.mask_fcs.append(
                nn.Linear(in_channels, in_channels, bias_attr=False))
            self.mask_fcs.append(nn.LayerNorm(in_channels))
            self.mask_fcs.append(act_fn())

        self.fc_mask = nn.Linear(in_channels, out_channels)

    def forward(self, x, proposal_feat, mask_preds, mask_shape=None):
        N, num_proposals = proposal_feat.shape[:2]
        if self.feat_transform is not None:
            x = self.feat_transform(x)

        C, H, W = x.shape[-3:]

        mask_h, mask_w = mask_preds.shape[-2:]
        if mask_h != H or mask_w != W:
            gather_mask = F.interpolate(mask_preds, (H, W),
                                        align_corners=False,
                                        mode='bilinear')
        else:
            gather_mask = mask_preds

        sigmoid_masks = F.softmax(gather_mask, 1)

        # Group Feature Assembling. Eq.(3) in original paper.
        x_feat = paddle.einsum('bnhw,bchw->bnc', sigmoid_masks, x)

        # obj_feat in shape [B, N, C, K, K] -> [B, N, C, K*K] -> [B, N, K*K, C]
        proposal_feat = proposal_feat.reshape(
            [N, num_proposals, self.in_channels, -1]).transpose([0, 1, 3, 2])
        obj_feat = self.kernel_update_conv(x_feat, proposal_feat)

        # [B, N, K*K, C] -> [B, N, K*K*C] -> [N, B, K*K*C]
        obj_feat = obj_feat.reshape([N, num_proposals, -1]).transpose([1, 0, 2])
        obj_feat = self.attention_norm(self.attention(obj_feat))
        # [N, B, K*K*C] -> [B, N, K*K*C]
        obj_feat = obj_feat.transpose([1, 0, 2])

        # obj_feat in shape [B, N, K*K*C] -> [B, N, K*K, C]
        obj_feat = obj_feat.reshape([N, num_proposals, -1, self.in_channels])

        # FFN
        if self.with_ffn:
            obj_feat = self.ffn_norm(self.ffn(obj_feat))

        mask_feat = obj_feat

        for reg_layer in self.mask_fcs:
            mask_feat = reg_layer(mask_feat)

        # [B, N, K*K, C] -> [B, N, C, K*K]
        mask_feat = self.fc_mask(mask_feat).transpose([0, 1, 3, 2])

        if (self.mask_transform_stride == 2 and self.feat_gather_stride == 1):
            mask_x = F.interpolate(x,
                                   scale_factor=0.5,
                                   mode='bilinear',
                                   align_corners=False)
            H, W = mask_x.shape[-2:]
        else:
            mask_x = x
        mask_feat = mask_feat.reshape(
            [N, num_proposals, C, self.conv_kernel_size, self.conv_kernel_size])
        # [B, C, H, W] -> [1, B*C, H, W]
        new_mask_preds = []
        for i in range(N):
            new_mask_preds.append(
                F.conv2d(mask_x[i:i + 1],
                         mask_feat[i],
                         padding=int(self.conv_kernel_size // 2)))

        new_mask_preds = paddle.concat(new_mask_preds, axis=0)
        new_mask_preds = new_mask_preds.reshape([N, num_proposals, H, W])
        if self.mask_transform_stride == 2:
            new_mask_preds = F.interpolate(new_mask_preds,
                                           scale_factor=2,
                                           mode='bilinear',
                                           align_corners=False)

        if mask_shape is not None and mask_shape[0] != H:
            new_mask_preds = F.interpolate(new_mask_preds,
                                           mask_shape,
                                           align_corners=False,
                                           mode='bilinear')

        return new_mask_preds, obj_feat.transpose([0, 1, 3, 2]).reshape([
            N, num_proposals, self.in_channels, self.conv_kernel_size,
            self.conv_kernel_size
        ])

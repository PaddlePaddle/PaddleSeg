# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils


@manager.MODELS.add_component
class PointRend(nn.Layer):
    """
    The SemanticFPN-PointRend implementation based on PaddlePaddle.

    The original article refers to
    Kirillov A, Wu Y, He K, et al. "PointRend: Image Segmentation As Rendering."
    (https://arxiv.org/abs/1912.08193).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): Backbone network, currently support Resnet50/101.
        backbone_indices (tuple, optional): Four values in the tuple indicate the indices of output of backbone.
        fpn_inplanes (list, optional): Input channels list(the feature channels from backbone) for lateral_conv constraction in FPN. Default: [256, 512, 1024, 2048].
        fpn_outplanes (int, optional): The output channels in FPN. Default: 256.
        point_num_fcs (int, optional): Number of fc layers in the head in PointHead. Default: 3.
        point_in_channels (list, optional): input channels of fc block in PointHead. Default: [256].
        point_out_channels (int, optional): Fc block's output channels in PointHead. Default: 256.
        point_in_index (list, optional): The indexs of input features to use in PointHead. Default: [0].
        point_num_points (int, optional): The number of point in training mode in PointHead. Default: 2048.
        point_oversample_ratio (int, optional): The sample ratio of points when in training mode in PointHead.
            sampled_point = num_points * oversample_ratio. Default: 3.
        point_importance_sample_ratio (float, optional): The importance sample ratio for compute num_uncertain_points in PointHead. Default: 0.75.
        point_scale_factor(int, optinal): The scale factor of F.interpolate in refine seg logits stage when in inference in PointHead. Default: 2.
        point_subdivision_steps(int, optional): Then refine steps in refine seg logits stage when in inference in PointHead. Default: 2.
        point_subdivision_num_points(int, optional): The points number for refine seg logits when in inference in PointHead. Default: 8196.
        point_dropout_ratio(float, optional): If the dropout_ratio >0, to use Dropout before output and the p of dropout is dropout_ratio in PointHead. Default: 0.1.
        point_coarse_pred_each_layer(bool, optional): Whether concatenate coarse feature with
            the output of each fc layer in PointHead. Default: True.
        point_conv_cfg(str): The config of Conv in PointHead. Default: 'Conv1D'.
        point_input_transform(str): The features transform method of inputs in PointHead.
            it can be found in function '_transform_inputs'. Defalut: 'multiple_select'.
        PFN_feature_strides(list): The strides for input feature maps and all strides suppose to be power of 2 in FPNHead. The first
            one is of largest resolution. Default: [4, 8, 16, 32].
        PFN_in_channels(list): The input feature's channels list in FPNHead. Default: [256, 256, 256, 256].
        PFN_channels(int,optional): The output channels of scale_head's Conv before Upsample block in FPNHead. Default: 128.
        PFN_in_index(list): The indexs of input features to use. it's shape should keep with in_channels in FPNHead. Default: [0, 1, 2, 3].
        PFN_dropout_ratio(float,optional): If the dropout_ratio >0, to use Dropout before output and the p of dropout is dropout_ratio in FPNHead. Default: 0.1.
        PFN_conv_cfg(str): The config of Conv. Default: 'Conv2D'.
        PFN_input_transform(str): The features transform method of inputs. it can be found in function '_transform_inputs' in FPNHead. Defalut: 'multiple_select'.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(
            self,
            num_classes,
            backbone,
            backbone_indices,
            fpn_inplanes=[256, 512, 1024, 2048],
            fpn_outplanes=256,
            point_in_channels=[256],
            point_out_channels=256,
            point_in_index=[0],
            point_num_fcs=3,
            point_num_points=2048,
            point_oversample_ratio=3,
            point_importance_sample_ratio=0.75,
            point_scale_factor=2,
            point_subdivision_steps=2,
            point_subdivision_num_points=8196,
            point_dropout_ratio=0,
            point_coarse_pred_each_layer=True,
            point_input_transform='multiple_select',  # resize_concat
            point_conv_cfg='Conv1D',
            PFN_feature_strides=[4, 8, 16, 32],
            PFN_in_channels=[256, 256, 256, 256],
            PFN_channels=128,
            PFN_in_index=[0, 1, 2, 3],
            PFN_dropout_ratio=0,
            PFN_conv_cfg='Conv2D',
            PFN_input_transform='multiple_select',
            align_corners=False,
            pretrained=None):
        super(PointRend, self).__init__()
        self.backbone = backbone
        self.backbone_indices = backbone_indices
        self.in_channels = [
            self.backbone.feat_channels[i] for i in backbone_indices
        ]

        self.neck = FPNNeck(
            fpn_inplanes=fpn_inplanes, fpn_outplanes=fpn_outplanes)
        self.pointhead = PointHead(
            in_channels=point_in_channels,
            out_channels=point_out_channels,
            num_classes=num_classes,
            in_index=point_in_index,
            num_fcs=point_num_fcs,
            num_points=point_num_points,
            oversample_ratio=point_oversample_ratio,
            importance_sample_ratio=point_importance_sample_ratio,
            scale_factor=point_scale_factor,
            subdivision_steps=point_subdivision_steps,
            subdivision_num_points=point_subdivision_num_points,
            dropout_ratio=point_dropout_ratio,
            align_corners=align_corners,
            coarse_pred_each_layer=point_coarse_pred_each_layer,
            input_transform=point_input_transform,  # resize_concat
            conv_cfg=point_conv_cfg)
        self.fpnhead = FPNHead(
            feature_strides=PFN_feature_strides,
            in_channels=PFN_in_channels,
            channels=PFN_channels,
            num_class=num_classes,
            in_index=PFN_in_index,
            dropout_ratio=PFN_dropout_ratio,
            conv_cfg=PFN_conv_cfg,
            input_transform=PFN_input_transform,
            align_corners=align_corners)

        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        feats = self.backbone(x)
        feats = [feats[i] for i in self.backbone_indices]
        fpn_feats = self.neck(feats)  # [n,256,64,128]*3 & [n,256,128,256]
        pfn_logits = self.fpnhead(
            fpn_feats
        )  # segmainoutput decode_head[0] 512*1024->[n, 19, 64, 128]
        point_logits = self.pointhead(
            fpn_feats, pfn_logits)  # segpointoutput decode_head[1]

        if self.training:
            logit_list = [
                F.interpolate(
                    logit,
                    paddle.shape(x)[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for logit in pfn_logits
            ]
            logit_list.append(point_logits)
        else:
            logit_list = [
                F.interpolate(
                    logit,
                    paddle.shape(x)[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for logit in point_logits
            ]
        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class PointHead(nn.Layer):
    """
    The PointHead implementation based on PaddlePaddle.

    PointHead use shared multi-layer perceptron (equivalent to
    nn.Conv1D) to predict the logit of input points. The fine-grained feature
    and coarse feature will be concatenate together for predication.

    The original article refers to:
    Kirillov A , Wu Y , He K , et al "PointRend: Image Segmentation As Rendering."
    (https://arxiv.org/abs/1912.08193)

    Args:
        num_classes (int): Number of classes for logits. Default: 19.
        num_fcs (int, optional): Number of fc layers in the head. Default: 3.
        in_channels (list): input channels of fc block. Default: [256].
        out_channels (int, optional): Fc block's output channels. Default: 256.
        in_index (list): The indexs of input features to use. Default: [0].
        num_points (int, optional): The number of point in training mode. Default: 2048.
        oversample_ratio (int, optional): The sample ratio of points when in training mode.
            sampled_point = num_points * oversample_ratio. Default: 3.
        importance_sample_ratio(float, optional): The importance sample ratio for compute num_uncertain_points. Default: 0.75.
        scale_factor(int, optional): The scale factor of F.interpolate in refine seg logits stage when in inference. Default: 2.
        subdivision_steps(int, optional): Then refine steps in refine seg logits stage when in inference. Default: 2.
        subdivision_num_points(int, optional): The points number for refine seg logits when in inference. Default: 8196.
        dropout_ratio(float, optional): If the dropout_ratio >0, to use Dropout before output and the p of dropout is dropout_ratio. Default: 0.1.
        coarse_pred_each_layer(bool, optional): Whether concatenate coarse feature with
            the output of each fc layer. Default: True.
        conv_cfg(str): The config of Conv. Default: 'Conv1D'.
        input_transform(str): The features transform method of inputs.
            it can be found in function '_transform_inputs'. Defalut: 'multiple_select'.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
    """

    def __init__(
            self,
            num_classes=19,
            num_fcs=3,
            in_channels=[256],
            out_channels=256,
            in_index=[0],
            num_points=2048,
            oversample_ratio=3,
            importance_sample_ratio=0.75,
            scale_factor=2,
            subdivision_steps=2,
            subdivision_num_points=8196,
            dropout_ratio=0.1,
            coarse_pred_each_layer=True,
            conv_cfg='Conv1D',
            input_transform='multiple_select',  # resize_concat
            align_corners=False):
        super(PointHead, self).__init__()

        self.in_channels = in_channels
        self.channels = out_channels
        self.in_index = in_index
        self.num_classes = num_classes
        self.num_fcs = num_fcs
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.scale_factor = scale_factor
        self.subdivision_steps = subdivision_steps
        self.subdivision_num_points = paddle.to_tensor(subdivision_num_points, dtype="int32")
        self.dropout_ratio = dropout_ratio
        self.coarse_pred_each_layer = coarse_pred_each_layer
        self.align_corners = align_corners
        self.input_transform = input_transform

        fc_in_channels = sum(self.in_channels) + self.num_classes
        fc_channels = self.channels
        self.fcs = nn.LayerList()
        for k in range(num_fcs):
            fc = ConvModule(
                fc_in_channels,
                fc_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
            )
            self.fcs.append(fc)
            fc_in_channels = fc_channels
            fc_in_channels += self.num_classes if self.coarse_pred_each_layer else 0
        self.fc_seg = nn.Conv1D(
            fc_in_channels,
            self.num_classes,
            kernel_size=1,
            stride=1,
            padding=0)

        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout(self.dropout_ratio)
        else:
            self.dropout = None

    def cls_seg(self, feat):
        """Classify each pixel with fc."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.fc_seg(feat)
        return output

    def _get_fine_grained_point_feats(self, x, points):
        """
        Sample from fine grained features.

        Args:
            x (list[Tensor]): Feature pyramid from by neck or backbone.
            points (Tensor): Point coordinates, shape (batch_size,
                num_points, 2).
        Returns:
            fine_grained_feats (Tensor): Sampled fine grained feature,
                shape (batch_size, sum(channels of x), num_points).
        """

        fine_grained_feats_list = [
            point_sample(_, points, align_corners=self.align_corners) for _ in x
        ]
        if len(fine_grained_feats_list) > 1:
            fine_grained_feats = paddle.concat(fine_grained_feats_list, axis=1)
        else:
            fine_grained_feats = fine_grained_feats_list[0]
        return fine_grained_feats

    def _get_coarse_point_feats(self, prev_output, points):
        """
        Sample from fine grained features.

        Args:
            prev_output (list[Tensor]): Prediction of previous decode head.
            points (Tensor): Point coordinates, shape (batch_size,
                num_points, 2).
        Returns:
            coarse_feats (Tensor): Sampled coarse feature, shape (batch_size,
                num_classes, num_points).
        """

        coarse_feats = point_sample(
            prev_output, points, align_corners=self.align_corners)
        return coarse_feats

    def _transform_inputs(self, inputs):
        """
        Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                F.interpolate(
                    x,
                    size=paddle.shape(inputs[0])[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = paddle.concat(upsampled_inputs, axis=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index[0]]
        return inputs

    def get_points_train(self, seg_logits, uncertainty_func):  # finish
        """
        Sample points for training.
        Sample points in [0, 1] x [0, 1] coordinate space based on their
        uncertainty. The uncertainties are calculated for each point using
        'uncertainty_func' function that takes point's logit prediction as
        input.

        Args:
            seg_logits (Tensor): Semantic segmentation logits, shape (
                batch_size, num_classes, height, width).
            uncertainty_func (func): uncertainty calculation function.
            cfg (dict): Training config of point head.
        Returns:
            point_coords (Tensor): A tensor of shape (batch_size, num_points,
                2) that contains the coordinates of ``num_points`` sampled
                points.
        """

        num_points = self.num_points
        oversample_ratio = self.oversample_ratio
        importance_sample_ratio = self.importance_sample_ratio
        assert oversample_ratio >= 1
        assert 0 <= importance_sample_ratio <= 1
        batch_size = paddle.shape(seg_logits)[0]
        num_sampled = int(num_points * oversample_ratio)
        point_coords = paddle.rand([batch_size, num_sampled, 2])
        point_logits = point_sample(seg_logits, point_coords)
        # It is crucial to calculate uncertainty based on the sampled
        # prediction value for the points. Calculating uncertainties of the
        # coarse predictions first and sampling them for points leads to
        # incorrect results.  To illustrate this: assume uncertainty func(
        # logits)=-abs(logits), a sampled point between two coarse
        # predictions with -1 and 1 logits has 0 logits, and therefore 0
        # uncertainty value. However, if we calculate uncertainties for the
        # coarse predictions first, both will have -1 uncertainty,
        # and sampled point will get -1 uncertainty.
        point_uncertainties = uncertainty_func(point_logits)
        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points
        idx = paddle.topk(
            point_uncertainties[:, 0, :], k=num_uncertain_points, axis=1)[1]
        shift = num_sampled * paddle.arange(batch_size, dtype='int64')
        idx += shift.unsqueeze([-1])
        idx = idx.reshape([-1])
        point_coords = paddle.index_select(
            point_coords.reshape([-1, 2]), idx, axis=0)
        point_coords = point_coords.reshape(
            [batch_size, num_uncertain_points, 2])
        if num_random_points > 0:
            rand_point_coords = paddle.rand([batch_size, num_random_points, 2])
            point_coords = paddle.concat((point_coords, rand_point_coords),
                                         axis=1)
        return point_coords

    def get_points_test(self, seg_logits, uncertainty_func):  # finish
        """
        Sample points for testing.
        Find ``num_points`` most uncertain points from ``uncertainty_map``.

        Args:
            seg_logits (Tensor): A tensor of shape (batch_size, num_classes,
                height, width) for class-specific or class-agnostic prediction.
            uncertainty_func (func): uncertainty calculation function.
            cfg (dict): Testing config of point head.
        Returns:
            point_indices (Tensor): A tensor of shape (batch_size, num_points)
                that contains indices from [0, height x width) of the most
                uncertain points.
            point_coords (Tensor): A tensor of shape (batch_size, num_points,
                2) that contains [0, 1] x [0, 1] normalized coordinates of the
                most uncertain points from the ``height x width`` grid .
        """

        num_points = self.subdivision_num_points
        uncertainty_map = uncertainty_func(seg_logits)
        batch_size = paddle.shape(uncertainty_map)[0]
        height = paddle.shape(uncertainty_map)[2]
        width = paddle.shape(uncertainty_map)[3]
        h_step = 1.0 / height
        w_step = 1.0 / width

        uncertainty_map = uncertainty_map.reshape([batch_size, height * width])
        num_points = paddle.min(paddle.concat([height * width, num_points]))
        point_indices = paddle.topk(uncertainty_map, num_points, axis=1)[1]
        point_coords = paddle.zeros([batch_size, num_points, 2],
                                    dtype='float32')
        point_coords[:, :, 0] = w_step / 2.0 + (
            point_indices % width).astype('float32') * w_step
        point_coords[:, :, 1] = h_step / 2.0 + (
            point_indices // width).astype('float32') * h_step
        return point_indices, point_coords

    def scatter_paddle(self, refined_seg_logits, point_indices, point_logits):
        """
        paddle version scatter : equal to pytorch version scatter(-1,point_indices,point_logits).

        Args:
            refined_seg_logits(Tensor): shape=[batch_size, channels, height * width]
            point_indices(Tensor): shape=[batch_size, channels, height * width]
            point_logits(Tensor): shape[batch_size, channels, height * width]
        Returns:
            scattered refined_seg_logits(Tensor).
        """

        original_shape = paddle.shape(refined_seg_logits)  # [batch_size, channels, height * width]
        new_refined_seg_logits = refined_seg_logits.flatten(0, 1)  # [N*C,H*W]
        offsets = (paddle.arange(paddle.shape(new_refined_seg_logits)[0]) *
                   paddle.shape(new_refined_seg_logits)[1]).unsqueeze(-1)  # [N*C,1]
        point_indices = point_indices.flatten(0, 1)  # [N*C,H*W]
        new_point_indices = (point_indices + offsets).flatten()
        point_logits = point_logits.flatten()  # [N*C*H*W]
        refined_seg_logits = paddle.scatter(
            refined_seg_logits.flatten(),
            new_point_indices,
            point_logits,
            overwrite=True)
        return refined_seg_logits.reshape(shape=original_shape)

    def forward_train(self, x, prev_output):
        with paddle.no_grad():
            points = self.get_points_train(prev_output, calculate_uncertainty)

        fine_grained_point_feats = self._get_fine_grained_point_feats(
            x, points)  # [2, 256, 2048]
        coarse_point_feats = self._get_coarse_point_feats(
            prev_output, points)  # [2, 19, 2048]
        # forward for train
        fusion_point_feats = paddle.concat(
            [fine_grained_point_feats, coarse_point_feats], axis=1)
        for fc in self.fcs:
            fusion_point_feats = fc(fusion_point_feats)
            if self.coarse_pred_each_layer:
                fusion_point_feats = paddle.concat(
                    (fusion_point_feats, coarse_point_feats), axis=1)
        point_logits = self.cls_seg(fusion_point_feats)
        return [point_logits, points]  # for points loss

    def forward(self, inputs, prev_output):
        """
        Forward function.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            prev_output (Tensor): The output of previous decode head.
        Returns:
            [point_logits,points]: For points loss when in training.
            [refined_seg_logits]: Output refined seg logits when in inference.
        """

        prev_output = prev_output[0]
        x = self._transform_inputs(inputs)
        if self.training:
            return self.forward_train(x, prev_output)
        else:
            refined_seg_logits = prev_output.clone()
            for _ in range(self.subdivision_steps):
                refined_seg_logits = F.interpolate(
                    refined_seg_logits,
                    scale_factor=self.scale_factor,
                    mode='bilinear',
                    align_corners=self.align_corners)

                save_shape = paddle.shape(refined_seg_logits)
                point_indices, points = self.get_points_test(
                    refined_seg_logits, calculate_uncertainty)
                fine_grained_point_feats = self._get_fine_grained_point_feats(
                    x, points)
                coarse_point_feats = self._get_coarse_point_feats(
                    prev_output, points)
                # forward for inference
                fusion_point_feats = paddle.concat(
                    [fine_grained_point_feats, coarse_point_feats], axis=1)
                for fc in self.fcs:
                    fusion_point_feats = fc(fusion_point_feats)
                    if self.coarse_pred_each_layer:
                        fusion_point_feats = paddle.concat(
                            (fusion_point_feats, coarse_point_feats), axis=1)
                point_logits = self.cls_seg(fusion_point_feats)
                point_indices = paddle.unsqueeze(point_indices, axis=1)
                point_indices = paddle.expand(point_indices, [-1, save_shape[1], -1])

                refined_seg_logits = paddle.flatten(refined_seg_logits, 2)
                refined_seg_logits = self.scatter_paddle(
                    refined_seg_logits, point_indices,
                    point_logits)  # 2->height * width dim
                refined_seg_logits = refined_seg_logits.reshape(save_shape)
            return [refined_seg_logits]


class FPNHead(nn.Layer):
    """
    This head is the implementation of Semantic FPN in paddle.

    The original article refers to:
    Kirillov, A. , et al. "Panoptic Feature Pyramid Networks."
    (https://arxiv.org/abs/1901.02446)

    Args:
        num_classes(int): The unique number of target classes. Default: 19.
        feature_strides(list): The strides for input feature maps and all strides suppose to be power of 2. The first
            one is of largest resolution. Default: [4, 8, 16, 32].
        in_channels(list): The input feature's channels list. Default: [256, 256, 256, 256].
        channels(int, optional): The output channels of scale_head's Conv before Upsample block. Default: 128.
        in_index(list): The indexs of input features to use. it's shape should keep with in_channels. Default: [0, 1, 2, 3].
        dropout_ratio(float, optional): If the dropout_ratio >0, to use Dropout before output and the p of dropout is dropout_ratio. Default: 0.1.
        conv_cfg(str): The config of Conv. Default: 'Conv2D'.
        input_transform(str): The features transform method of inputs. it can be found in function '_transform_inputs'. Defalut: 'multiple_select'.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
    """

    def __init__(
            self,
            num_class=19,
            feature_strides=[4, 8, 16, 32],
            in_channels=[256, 256, 256, 256],
            channels=128,
            in_index=[0, 1, 2, 3],
            dropout_ratio=0.1,
            conv_cfg='Conv2D',
            input_transform='multiple_select',
            align_corners=False,
    ):
        super(FPNHead, self).__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.in_channels = in_channels
        self.channels = channels
        self.in_index = in_index
        self.num_class = num_class
        self.conv_cfg = conv_cfg
        self.dropout_ratio = dropout_ratio
        self.input_transform = input_transform
        self.align_corners = align_corners
        self.scale_heads = nn.LayerList()

        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))

        self.conv_seg = nn.Conv2D(self.channels, self.num_class, kernel_size=1)

        if self.dropout_ratio is not None:
            self.dropout = nn.Dropout2D(self.dropout_ratio)
        else:
            self.dropout = None

    def cls_seg(self, feat):
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def _transform_inputs(self, inputs):
        """
        Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                F.interpolate(
                    x,
                    size=paddle.shape(inputs[0])[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = paddle.concat(upsampled_inputs, axis=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index[0]]

        return inputs

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            output = output + F.interpolate(
                self.scale_heads[i](x[i]),
                size=paddle.shape(output)[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        output = self.cls_seg(output)
        return [output]


class FPNNeck(nn.Layer):
    """
    The FPN Neck implementation in paddle.

    Args:
        fpn_inplanes (list, optional): Input channels list(the feature channels from backbone) for lateral_conv constraction. Default: [256, 512, 1024, 2048].
        fpn_outplanes (int, optional): The output channels. Default: 256.
    """

    def __init__(
            self,
            fpn_inplanes=[256, 512, 1024, 2048],
            fpn_outplanes=256,
    ):
        super(FPNNeck, self).__init__()
        self.lateral_convs = []
        self.fpn_out = []

        # FPN head
        for fpn_inplane in fpn_inplanes:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2D(fpn_inplane, fpn_outplanes, 1),
                    layers.SyncBatchNorm(fpn_outplanes), nn.ReLU()))
            self.fpn_out.append(
                nn.Sequential(
                    layers.ConvBNReLU(
                        fpn_outplanes, fpn_outplanes, 3, bias_attr=False)))

        self.lateral_convs = nn.LayerList(self.lateral_convs)
        self.fpn_out = nn.LayerList(self.fpn_out)

    def forward(self, conv_out):
        last_out = self.lateral_convs[-1](conv_out[-1])
        f = last_out
        fpn_feature_list = [last_out]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.lateral_convs[i](conv_x)
            prev_shape = paddle.shape(conv_x)[2:]
            f = conv_x + F.interpolate(
                f, prev_shape, mode='bilinear', align_corners=True)
            fpn_feature_list.append(self.fpn_out[i](f))
        return fpn_feature_list


class ConvModule(nn.Layer):
    """
    ConvModule includes Conv1/Conv2D.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=0,
                 stride=1,
                 conv_cfg='Conv1D',
                 norm_cfg='None',
                 **kwargs):
        super().__init__()
        if (conv_cfg == 'Conv1D'):
            self._conv = nn.Conv1D(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                **kwargs)
        if (conv_cfg == 'Conv2D'):
            self._conv = nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                **kwargs)
        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        if (norm_cfg != 'None'):
            self._batch_norm = layers.SyncBatchNorm(
                out_channels, data_format=data_format)
        else:
            self._batch_norm = None

    def forward(self, x):
        x = self._conv(x)
        if (self._batch_norm != None):
            x = self._batch_norm(x)
        x = F.relu(x)
        return x


class Upsample(nn.Layer):
    """
    Upsample Module.
    """

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if not self.size:
            return F.interpolate(x, None, self.scale_factor, self.mode, self.align_corners)
        else:
            return F.interpolate(x, self.size, None, self.mode, self.align_corners)


def point_sample(input, points, align_corners=False, **kwargs):
    """
    A wrapper around :func:`grid_sample` to support 3D point_coords tensors
    Unlike :func:`torch.nn.functional.grid_sample` it assumes point_coords to
    lie inside ``[0, 1] x [0, 1]`` square.

    Args:
        input (Tensor): Feature map, shape (N, C, H, W).
        points (Tensor): Image based absolute point coordinates (normalized),
            range [0, 1] x [0, 1], shape (N, P, 2) or (N, Hgrid, Wgrid, 2).
        align_corners (bool): Whether align_corners. Default: False
    Returns:
        Tensor: Features of `point` on `input`, shape (N, C, P) or
            (N, C, Hgrid, Wgrid).
    """

    def denormalize(grid):
        """Denormalize input grid from range [0, 1] to [-1, 1]
        Args:
            grid (Tensor): The grid to be denormalize, range [0, 1].
        Returns:
            Tensor: Denormalized grid, range [-1, 1].
        """
        return grid * 2.0 - 1.0

    add_dim = False
    if points.dim() == 3:
        add_dim = True
        points = paddle.unsqueeze(points, axis=2)
    output = F.grid_sample(
        input, denormalize(points), align_corners=align_corners, **kwargs)
    if add_dim:
        output = paddle.squeeze(output, axis=3)
    return output


def calculate_uncertainty(seg_logits):
    """
    Estimate uncertainty based on seg logits.
    For each location of the prediction ``seg_logits`` we estimate
    uncertainty as the difference between top first and top second
    predicted logits.

    Args:
        seg_logits (Tensor): Semantic segmentation logits,
            shape (batch_size, num_classes, height, width).
    Returns:
        scores (Tensor): T uncertainty scores with the most uncertain
            locations having the highest uncertainty score, shape (
            batch_size, 1, height, width)
    """

    top2_scores = paddle.topk(seg_logits, k=2, axis=1)[0]
    return paddle.unsqueeze(top2_scores[:, 1] - top2_scores[:, 0], axis=1)

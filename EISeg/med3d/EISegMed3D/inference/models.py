import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "./.."))

import paddle
import paddle.nn as nn
import numpy as np

from inference.ops import DistMaps3D, ScaleLayer, BatchImageNormalize3D


class ISModel3D(nn.Layer):

    def __init__(
        self,
        use_rgb_conv=True,
        with_aux_output=False,
        norm_radius=2,
        use_disks=False,
        cpu_dist_maps=False,
        clicks_groups=None,
        with_prev_mask=False,  # True
        use_leaky_relu=False,
        binary_prev_mask=False,
        conv_extend=False,
        norm_layer=nn.BatchNorm3D,
        norm_mean_std=(
            [
                0.00040428873,
            ],
            [
                0.00059983705,
            ],
        ),
    ):  #  image.std(): [0.00053328] image.mean() [0.00023692])
        super().__init__()

        self.with_aux_output = with_aux_output
        self.clicks_groups = clicks_groups
        self.with_prev_mask = with_prev_mask
        self.binary_prev_mask = binary_prev_mask
        self.normalization = BatchImageNormalize3D(norm_mean_std[0],
                                                   norm_mean_std[1])

        self.coord_feature_ch = 2
        if clicks_groups is not None:
            self.coord_feature_ch *= len(clicks_groups)

        if self.with_prev_mask:
            self.coord_feature_ch += 1  # 3

        if use_rgb_conv:
            rgb_conv_layers = [
                nn.Conv3D(in_channels=1 + self.coord_feature_ch,
                          out_channels=6 + self.coord_feature_ch,
                          kernel_size=1),
                norm_layer(6 + self.coord_feature_ch),
                nn.LeakyReLU(
                    negative_slope=0.2) if use_leaky_relu else nn.ReLU(),
                nn.Conv3D(in_channels=6 + self.coord_feature_ch,
                          out_channels=1,
                          kernel_size=1),
            ]
            self.rgb_conv = nn.Sequential(*rgb_conv_layers)

        elif conv_extend:
            self.rgb_conv = None
            self.maps_transform = nn.Conv3D(in_channels=self.coord_feature_ch,
                                            out_channels=64,
                                            kernel_size=3,
                                            stride=2,
                                            padding=1)
        else:
            self.rgb_conv = None
            mt_layers = [
                nn.Conv3D(in_channels=self.coord_feature_ch,
                          out_channels=16,
                          kernel_size=1),
                nn.LeakyReLU(
                    negative_slope=0.2) if use_leaky_relu else nn.ReLU(),
                nn.Conv3D(in_channels=16,
                          out_channels=16,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                ScaleLayer(init_value=0.05, lr_mult=1),
            ]
            self.maps_transform = nn.Sequential(*mt_layers)
        if self.clicks_groups is not None:
            self.dist_maps = nn.LayerList()
            for click_radius in self.clicks_groups:
                self.dist_maps.append(
                    DistMaps3D(norm_radius=click_radius,
                               spatial_scale=1.0,
                               cpu_mode=cpu_dist_maps,
                               use_disks=use_disks))
        else:
            self.dist_maps = DistMaps3D(norm_radius=norm_radius,
                                        spatial_scale=1.0,
                                        cpu_mode=cpu_dist_maps,
                                        use_disks=use_disks)

    def forward(self, image, coord_features):
        if self.rgb_conv is not None:
            x = self.rgb_conv(paddle.concat((image, coord_features),
                                            axis=1))  # [B, 4, H, W, D] #
            outputs = self.backbone_forward(x)
        else:
            coord_features = self.maps_transform(
                coord_features)  # [B, 3, H, W, D]
            outputs = self.backbone_forward(image, coord_features)

        outputs["instances"] = nn.functional.interpolate(
            outputs["instances"],
            size=image.shape[2:],  # [4, 20, 512, 512, 12]
            mode="trilinear",
            align_corners=True,
            data_format="NCDHW",
        )  # image [4  , 1  , 512, 512, 12 ]
        if self.with_aux_output:
            outputs["instances_aux"] = nn.functional.interpolate(
                outputs["instances_aux"],
                size=image.shape[2:],
                mode="biltrilinearinear",
                align_corners=True,
                data_format="NCDHW",
            )
        return outputs

    def prepare_input(self, image):
        prev_mask = None
        if self.with_prev_mask:
            prev_mask = paddle.slice(
                image,
                axes=[
                    1,
                ],
                starts=[
                    1,
                ],
                ends=[
                    1000,
                ],
            )
            image = paddle.slice(
                image,
                axes=[
                    1,
                ],
                starts=[
                    0,
                ],
                ends=[
                    1,
                ],
            )
            # prev_mask = image[:, 1:, :, :, :]
            # image = image[:, :1, :, :, :]
            if self.binary_prev_mask:
                prev_mask = (prev_mask > 0.5).astype("float32")

        image = self.normalization(image)  # why?
        return image, prev_mask

    def backbone_forward(self, image, coord_features=None):
        raise NotImplementedError

    def get_coord_features(self, image, prev_mask, points):

        coord_features = self.dist_maps(
            image,
            points)  #  [16, 1, 512, 512, 12], [16, 48, 4]. # [B, 2, H, W, D]

        if prev_mask is not None:
            coord_features = paddle.concat((prev_mask, coord_features),
                                           axis=1)  # [B, 3, H, W, D]

        return coord_features


def split_points_by_order(tpoints,
                          groups):  # todo check if point have dimension problem
    points = tpoints.numpy()
    num_groups = len(groups)
    bs = points.shape[0]
    num_points = points.shape[1] // 2

    groups = [x if x > 0 else num_points for x in groups]
    group_points = [
        np.full((bs, 2 * x, 3), -1, dtype=np.float32) for x in groups
    ]

    last_point_indx_group = np.zeros((bs, num_groups, 2), dtype=np.int)
    for group_indx, group_size in enumerate(groups):
        last_point_indx_group[:, group_indx, 1] = group_size

    for bindx in range(bs):
        for pindx in range(2 * num_points):
            point = points[bindx, pindx, :]
            group_id = int(point[2])
            if group_id < 0:
                continue

            is_negative = int(pindx >= num_points)
            if group_id >= num_groups or (group_id == 0 and is_negative
                                          ):  # disable negative first click
                group_id = num_groups - 1

            new_point_indx = last_point_indx_group[bindx, group_id, is_negative]
            last_point_indx_group[bindx, group_id, is_negative] += 1

            group_points[group_id][bindx, new_point_indx, :] = point

    group_points = [
        paddle.to_tensor(x, dtype=tpoints.dtype) for x in group_points
    ]

    return group_points


from paddleseg.utils import utils


class LUConv(nn.Layer):

    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = nn.ELU() if elu else nn.PReLU(nchan)
        self.conv1 = nn.Conv3D(nchan, nchan, kernel_size=5, padding=2)

        self.bn1 = nn.BatchNorm3D(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))

        return out


def _make_nConv(nchan, depth, elu):
    """
    Make depth number of layer(convbnrelu) and don't change the channel
    Add Nonlinearity into the network
    """
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Layer):
    """
    Transfer the input into 16 channels + tiled input
    """

    def __init__(self, in_channels, elu):
        super(InputTransition, self).__init__()
        self.num_features = 16
        self.in_channels = in_channels

        self.conv1 = nn.Conv3D(self.in_channels,
                               self.num_features,
                               kernel_size=5,
                               padding=2)

        self.bn1 = nn.BatchNorm3D(self.num_features)

        self.relu1 = nn.ELU() if elu else nn.PReLU(self.num_features)

    def forward(self, x):
        out = self.conv1(x)
        repeat_rate = int(self.num_features / self.in_channels)
        out = self.bn1(out)
        x_tile = x.tile([1, repeat_rate, 1, 1, 1])
        return self.relu1(paddle.add(out, x_tile))


class DownTransition(nn.Layer):

    def __init__(self,
                 inChans,
                 nConvs,
                 elu,
                 dropout=False,
                 downsample_stride=(2, 2, 2),
                 kernel=(2, 2, 2)):
        """
        1. double the output channel and downsample the input using down_conv(the kernel size can be changed)
        2. add dropout by option
        3. add nConvs layer to add linearity and add with original downsample one
        """
        super(DownTransition, self).__init__()
        outChans = 2 * inChans
        self.if_dropout = dropout
        self.down_conv = nn.Conv3D(inChans,
                                   outChans,
                                   kernel_size=kernel,
                                   stride=downsample_stride)
        self.bn1 = nn.BatchNorm3D(outChans)
        self.relu1 = nn.ELU() if elu else nn.PReLU(outChans)
        self.relu2 = nn.ELU() if elu else nn.PReLU(outChans)
        self.dropout = nn.Dropout3D()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.dropout(down) if self.if_dropout else down
        out = self.ops(out)
        out = paddle.add(out, down)
        out = self.relu2(out)

        return out


class UpTransition(nn.Layer):

    def __init__(
            self,
            inChans,
            outChans,
            nConvs,
            elu,
            dropout=False,
            dropout2=False,
            upsample_stride_size=(2, 2, 2),
            kernel=(2, 2, 2),
    ):
        super(UpTransition, self).__init__()
        """
        1. Add dropout to input and skip input optionally (generalization)
        2. Use Conv3DTranspose to upsample (upsample)
        3. concate the upsampled and skipx (multi-leval feature fusion)
        4. Add nConvs convs and residually add with result of step(residual + nonlinearity)
        """
        self.up_conv = nn.Conv3DTranspose(inChans,
                                          outChans // 2,
                                          kernel_size=kernel,
                                          stride=upsample_stride_size)

        self.bn1 = nn.BatchNorm3D(outChans // 2)
        self.relu1 = nn.ELU() if elu else nn.PReLU(outChans // 2)
        self.relu2 = nn.ELU() if elu else nn.PReLU(outChans)
        self.if_dropout = dropout
        self.if_dropout2 = dropout2
        self.dropout1 = nn.Dropout3D()
        self.dropout2 = nn.Dropout3D()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.dropout1(x) if self.if_dropout else x
        skipx = self.dropout2(skipx) if self.if_dropout2 else skipx
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = paddle.concat((out, skipx), 1)
        out = self.ops(xcat)
        out = self.relu2(paddle.add(out, xcat))

        return out


class OutputTransition(nn.Layer):

    def __init__(self, in_channels, num_classes, elu):
        """
        conv the output down to channels as the desired classesv
        """
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3D(in_channels,
                               num_classes,
                               kernel_size=5,
                               padding=2)
        self.bn1 = nn.BatchNorm3D(num_classes)
        self.relu1 = nn.ELU() if elu else nn.PReLU(num_classes)

        self.conv2 = nn.Conv3D(num_classes, num_classes, kernel_size=1)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return out


class VNet(nn.Layer):
    """
    Implementations based on the Vnet paper: https://arxiv.org/abs/1606.04797
    """

    def __init__(
            self,
            elu=False,
            in_channels=1,
            num_classes=2,
            pretrained=None,
            kernel_size=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
            stride_size=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
    ):
        super().__init__()
        self.best_loss = 1000000
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.in_tr = InputTransition(in_channels, elu=elu)
        self.down_tr32 = DownTransition(16,
                                        1,
                                        elu,
                                        downsample_stride=stride_size[0],
                                        kernel=kernel_size[0])
        self.down_tr64 = DownTransition(32,
                                        2,
                                        elu,
                                        downsample_stride=stride_size[1],
                                        kernel=kernel_size[1])
        self.down_tr128 = DownTransition(64,
                                         3,
                                         elu,
                                         dropout=True,
                                         downsample_stride=stride_size[2],
                                         kernel=kernel_size[2])
        self.down_tr256 = DownTransition(128,
                                         2,
                                         elu,
                                         dropout=True,
                                         downsample_stride=stride_size[3],
                                         kernel=kernel_size[3])
        self.up_tr256 = UpTransition(256,
                                     256,
                                     2,
                                     elu,
                                     dropout=True,
                                     dropout2=True,
                                     upsample_stride_size=stride_size[3],
                                     kernel=kernel_size[3])
        self.up_tr128 = UpTransition(256,
                                     128,
                                     2,
                                     elu,
                                     dropout=True,
                                     dropout2=True,
                                     upsample_stride_size=stride_size[2],
                                     kernel=kernel_size[2])
        self.up_tr64 = UpTransition(128,
                                    64,
                                    1,
                                    elu,
                                    upsample_stride_size=stride_size[1],
                                    kernel=kernel_size[1])
        self.up_tr32 = UpTransition(64,
                                    32,
                                    1,
                                    elu,
                                    upsample_stride_size=stride_size[0],
                                    kernel=kernel_size[0])
        self.out_tr = OutputTransition(32, num_classes, elu)

        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x, additional_features):  # [4, 1, 512, 512, 12]
        x = self.in_tr(x)  # dropout cause a lot align problem

        if additional_features is not None:  # todo check shape [B, 16, H, W, D] # [4, 16, 512, 512, 12] #
            x = x + additional_features

        out32 = self.down_tr32(x)  # [4, 32, 256, 256, 9]
        out64 = self.down_tr64(out32)  # [4, 64, 128, 128, 8]
        out128 = self.down_tr128(out64)  # [4, 128, 64, 64, 4]
        out256 = self.down_tr256(out128)  # [4, 256, 32, 32, 2]
        out = self.up_tr256(out256, out128)  # [4, 256, 64, 64, 4]
        out = self.up_tr128(out, out64)  # [4, 128, 128, 128, 8]
        out = self.up_tr64(out, out32)  # [4, 64, 256, 256, 9]
        out = self.up_tr32(out, x)  # [4, 32, 512, 512, 12]
        out = self.out_tr(out)  # [4, num_classes, 512, 512, 12]
        return out


class VNetModel(ISModel3D):
    # @serialize
    def __init__(self,
                 elu=False,
                 in_channels=1,
                 num_classes=2,
                 pretrained=None,
                 kernel_size=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
                 stride_size=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
                 norm_layer=nn.BatchNorm3D,
                 **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)

        self.feature_extractor = VNet(
            elu=elu,
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
            kernel_size=kernel_size,
            stride_size=stride_size,
        )  # diff: 去除了backbone mult，因为没有backbone

    def backbone_forward(self, image, coord_features=None):
        backbone_features = self.feature_extractor(
            image, coord_features)  # todo ：增加对点特征的融合

        return {
            "instances": backbone_features,
            "instances_aux": backbone_features,
        }  # result: 直接输出最后多少类别的分类tensor  # [4, num_classes , 512, 512, 12]

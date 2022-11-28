# This file is heavily based on https://github.com/czczup/ViT-Adapter

import math
import warnings
from functools import partial

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models.backbones.transformer_utils import DropPath
from paddleseg.models.layers.ms_deformable_attention import MSDeformAttn


def get_reference_points(spatial_shapes):
    reference_points_list = []
    for _, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = paddle.meshgrid(
            paddle.linspace(
                0.5, H_ - 0.5, H_, dtype='float32'),
            paddle.linspace(
                0.5, W_ - 0.5, W_, dtype='float32'))
        ref_y = ref_y.reshape([1, -1]) / H_
        ref_x = ref_x.reshape([1, -1]) / W_
        ref = paddle.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = paddle.concat(reference_points_list, 1)
    reference_points = paddle.unsqueeze(reference_points, axis=2)
    return reference_points


def deform_inputs(x):
    _, _, h, w = x.shape
    spatial_shapes = paddle.to_tensor(
        [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)],
        dtype='int64')
    level_start_index = paddle.concat((paddle.zeros(
        (1, ), dtype='int64'), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 16, w // 16)])
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = paddle.to_tensor([(h // 16, w // 16)], dtype='int64')
    level_start_index = paddle.concat((paddle.zeros(
        (1, ), dtype='int64'), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points(
        [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)])
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

    return deform_inputs1, deform_inputs2


class DWConv(nn.Layer):
    """
    The specific DWConv unsed in ConvFFN. 
    """

    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2D(dim, dim, 3, 1, 1, bias_attr=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose([0, 2, 1]).reshape(
            [B, C, H * 2, W * 2])
        x2 = x[:, 16 * n:20 * n, :].transpose([0, 2, 1]).reshape([B, C, H, W])
        x3 = x[:, 20 * n:, :].transpose([0, 2, 1]).reshape(
            [B, C, H // 2, W // 2])
        x1 = self.dwconv(x1).flatten(2).transpose([0, 2, 1])
        x2 = self.dwconv(x2).flatten(2).transpose([0, 2, 1])
        x3 = self.dwconv(x3).flatten(2).transpose([0, 2, 1])
        x = paddle.concat([x1, x2, x3], axis=1)
        return x


class ConvFFN(nn.Layer):
    """
    The implementation of ConvFFN unsed in Extractor.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Extractor(nn.Layer):
    """
    The Extractor module in ViT-Adapter.
    """

    def __init__(self,
                 dim,
                 num_heads=6,
                 n_points=4,
                 n_levels=1,
                 deform_ratio=1.0,
                 with_cffn=True,
                 cffn_ratio=0.25,
                 drop=0.,
                 drop_path=0.,
                 norm_layer=partial(
                     nn.LayerNorm, epsilon=1e-6)):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim,
            n_levels=n_levels,
            n_heads=num_heads,
            n_points=n_points,
            ratio=deform_ratio)
        self.with_cffn = with_cffn
        if with_cffn:
            self.ffn = ConvFFN(
                in_features=dim,
                hidden_features=int(dim * cffn_ratio),
                drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(
                drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes,
                level_start_index, H, W):
        attn = self.attn(
            self.query_norm(query), reference_points,
            self.feat_norm(feat), spatial_shapes, level_start_index, None)
        query = query + attn

        if self.with_cffn:
            query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
        return query


class Injector(nn.Layer):
    """
    The Injector module in ViT-Adapter.
    """

    def __init__(self,
                 dim,
                 num_heads=6,
                 n_points=4,
                 n_levels=1,
                 deform_ratio=1.0,
                 norm_layer=partial(
                     nn.LayerNorm, epsilon=1e-6),
                 init_values=0.):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim,
            n_levels=n_levels,
            n_heads=num_heads,
            n_points=n_points,
            ratio=deform_ratio)
        self.gamma = self.create_parameter(
            shape=(dim, ),
            default_initializer=paddle.nn.initializer.Constant(
                value=init_values))

    def forward(self, query, reference_points, feat, spatial_shapes,
                level_start_index):
        attn = self.attn(
            self.query_norm(query), reference_points,
            self.feat_norm(feat), spatial_shapes, level_start_index, None)
        return query + self.gamma * attn


class InteractionBlock(nn.Layer):
    """
    Combine the Extractor, Extractor and ViT Blocks.
    """

    def __init__(self,
                 dim,
                 num_heads=6,
                 n_points=4,
                 norm_layer=partial(
                     nn.LayerNorm, epsilon=1e-6),
                 drop=0.,
                 drop_path=0.,
                 with_cffn=True,
                 cffn_ratio=0.25,
                 init_values=0.,
                 deform_ratio=1.0,
                 extra_extractor=False):
        super().__init__()

        self.injector = Injector(
            dim=dim,
            n_levels=3,
            num_heads=num_heads,
            init_values=init_values,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio)
        self.extractor = Extractor(
            dim=dim,
            n_levels=1,
            num_heads=num_heads,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=drop,
            drop_path=drop_path)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(
                    dim=dim,
                    num_heads=num_heads,
                    n_points=n_points,
                    norm_layer=norm_layer,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    drop=drop,
                    drop_path=drop_path) for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W):
        x = self.injector(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2])

        for _, blk in enumerate(blocks):
            x = blk(x, H, W)

        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            H=H,
            W=W)

        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    H=H,
                    W=W)

        return x, c


class InteractionBlockWithCls(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=6,
                 n_points=4,
                 norm_layer=partial(
                     nn.LayerNorm, eps=1e-6),
                 drop=0.,
                 drop_path=0.,
                 with_cffn=True,
                 cffn_ratio=0.25,
                 init_values=0.,
                 deform_ratio=1.0,
                 extra_extractor=False):
        super().__init__()

        self.injector = Injector(
            dim=dim,
            n_levels=3,
            num_heads=num_heads,
            init_values=init_values,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio)
        self.extractor = Extractor(
            dim=dim,
            n_levels=1,
            num_heads=num_heads,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=drop,
            drop_path=drop_path)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(
                    dim=dim,
                    num_heads=num_heads,
                    n_points=n_points,
                    norm_layer=norm_layer,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    drop=drop,
                    drop_path=drop_path) for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, cls, blocks, deform_inputs1, deform_inputs2, H, W):
        x = self.injector(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2])
        x = paddle.concat((cls, x), axis=1)
        for _, blk in enumerate(blocks):
            x = blk(x, H, W)
        cls, x = x[:, :1, ], x[:, 1:, ]
        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            H=H,
            W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    H=H,
                    W=W)
        return x, c, cls


class SpatialPriorModule(nn.Layer):
    def __init__(self, inplanes=64, embed_dim=384):
        super().__init__()

        self.stem = nn.Sequential(*[
            nn.Conv2D(
                3,
                inplanes,
                kernel_size=3,
                stride=2,
                padding=1,
                bias_attr=False), nn.SyncBatchNorm(inplanes), nn.ReLU(),
            nn.Conv2D(
                inplanes,
                inplanes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False), nn.SyncBatchNorm(inplanes), nn.ReLU(),
            nn.Conv2D(
                inplanes,
                inplanes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False), nn.SyncBatchNorm(inplanes), nn.ReLU(),
            nn.MaxPool2D(
                kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2D(
                inplanes,
                2 * inplanes,
                kernel_size=3,
                stride=2,
                padding=1,
                bias_attr=False), nn.SyncBatchNorm(2 * inplanes), nn.ReLU()
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2D(
                2 * inplanes,
                4 * inplanes,
                kernel_size=3,
                stride=2,
                padding=1,
                bias_attr=False), nn.SyncBatchNorm(4 * inplanes), nn.ReLU()
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2D(
                4 * inplanes,
                4 * inplanes,
                kernel_size=3,
                stride=2,
                padding=1,
                bias_attr=False), nn.SyncBatchNorm(4 * inplanes), nn.ReLU()
        ])
        self.fc1 = nn.Conv2D(
            inplanes,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=True)
        self.fc2 = nn.Conv2D(
            2 * inplanes,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=True)
        self.fc3 = nn.Conv2D(
            4 * inplanes,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=True)
        self.fc4 = nn.Conv2D(
            4 * inplanes,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=True)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)

        bs, dim, _, _ = c1.shape
        c2 = c2.reshape([bs, dim, -1]).transpose([0, 2, 1])  # 8s
        c3 = c3.reshape([bs, dim, -1]).transpose([0, 2, 1])  # 16s
        c4 = c4.reshape([bs, dim, -1]).transpose([0, 2, 1])  # 32s

        return c1, c2, c3, c4

# This is heavily based on https://github.com/czczup/ViT-Adapter

import math
import warnings
from functools import partial

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models.backbones.transformer_utils import DropPath
from paddleseg.cvlibs.param_init import constant_init, xavier_uniform

import ms_deform_attn as msda  # first install ms_deform_attn


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
    bs, c, h, w = x.shape
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


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError('invalid input for _is_power_of_2: {} (type: {})'.
                         format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class ConvFFN(nn.Layer):
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


class DWConv(nn.Layer):
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


class MSDeformAttn(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_levels=4,
                 n_heads=8,
                 n_points=4,
                 ratio=1.0):
        """Multi-Scale Deformable Attention Module.

        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, '
                             'but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2
        # which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make "
                          'the dimension of each attention head a power of 2 '
                          'which is more efficient in our CUDA implementation.')

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.ratio = ratio
        self.sampling_offsets = nn.Linear(d_model,
                                          n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model,
                                           n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, int(d_model * ratio))
        self.output_proj = nn.Linear(int(d_model * ratio), d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_init(self.sampling_offsets.weight, value=0.)
        thetas = paddle.arange(
            self.n_heads, dtype='float32') * (2.0 * math.pi / self.n_heads)
        grid_init = paddle.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(
            -1, keepdim=True)[0]).reshape([self.n_heads, 1, 1, 2]).tile(
                [1, self.n_levels, self.n_points, 1])
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        with paddle.no_grad():
            grid_init = grid_init.reshape([-1])
            self.sampling_offsets.bias = self.create_parameter(
                shape=grid_init.shape,
                default_initializer=paddle.nn.initializer.Assign(grid_init))

        constant_init(self.attention_weights.weight, value=0.)
        constant_init(self.attention_weights.bias, value=0.)
        xavier_uniform(self.value_proj.weight)
        constant_init(self.value_proj.bias, value=0.)
        xavier_uniform(self.output_proj.weight)
        constant_init(self.output_proj.bias, value=0.)

    def forward(self,
                query,
                reference_points,
                input_flatten,
                input_spatial_shapes,
                input_level_start_index,
                input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """

        def masked_fill(x, mask, value):
            y = paddle.full(x.shape, value, x.dtype)
            return paddle.where(mask, y, x)

        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]
                ).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = masked_fill(value, input_padding_mask[..., None], float(0))

        value = value.reshape([
            N, Len_in, self.n_heads,
            int(self.ratio * self.d_model) // self.n_heads
        ])
        sampling_offsets = self.sampling_offsets(query).reshape(
            [N, Len_q, self.n_heads, self.n_levels, self.n_points, 2])
        attention_weights = self.attention_weights(query).reshape(
            [N, Len_q, self.n_heads, self.n_levels * self.n_points])
        attention_weights = F.softmax(attention_weights, -1).\
            reshape([N, Len_q, self.n_heads, self.n_levels, self.n_points])

        if reference_points.shape[-1] == 2:
            offset_normalizer = paddle.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]],
                -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'
                .format(reference_points.shape[-1]))
        output = msda.ms_deform_attn(
            value, input_spatial_shapes, input_level_start_index,
            sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output


class Extractor(nn.Layer):
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
        def _inner_forward(query, feat):
            attn = self.attn(
                self.query_norm(query), reference_points,
                self.feat_norm(feat), spatial_shapes, level_start_index, None)
            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(
                    self.ffn(self.ffn_norm(query), H, W))
            return query

        query = _inner_forward(query, feat)

        return query


class Injector(nn.Layer):
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
        def _inner_forward(query, feat):
            attn = self.attn(
                self.query_norm(query), reference_points,
                self.feat_norm(feat), spatial_shapes, level_start_index, None)
            return query + self.gamma * attn

        query = _inner_forward(query, feat)

        return query


class InteractionBlock(nn.Layer):
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
        debug = False
        x = self.injector(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2])
        if debug:
            print('x', x.cpu().numpy().mean())

        for idx, blk in enumerate(blocks):
            x = blk(x, H, W)
            if debug:
                print('x block_{}'.format(idx), x.cpu().numpy().mean())

        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            H=H,
            W=W)
        if debug:
            print('c', c.cpu().numpy().mean())

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
            if debug:
                print('c', c.cpu().numpy().mean())

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
        for idx, blk in enumerate(blocks):
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
        def _inner_forward(x):
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

        outs = _inner_forward(x)
        return outs

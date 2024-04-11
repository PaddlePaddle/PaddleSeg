import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers
from paddleseg.utils import utils, logger
from paddleseg.models.backbones.transformer_utils import *


class PadHelper:
    """ Make the size of feature map divisible by local group size."""

    def __init__(self, local_group_size=7):
        self.lgs = local_group_size
        if not isinstance(self.lgs, (tuple, list)):
            self.lgs = to_2tuple(self.lgs)
        assert len(self.lgs) == 2, "The length of self.lgs must be 2."

    def pad_if_needed(self, x, size):
        n, h, w, c = size
        pad_h = paddle.cast(
            paddle.ceil(h / self.lgs[0]) * self.lgs[0] - h, paddle.int32)
        pad_w = paddle.cast(
            paddle.ceil(w / self.lgs[1]) * self.lgs[1] - w, paddle.int32)
        if pad_h > 0 or pad_w > 0:  # center-pad the feature on H and W axes
            return F.pad(x,
                         paddle.to_tensor([
                             pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                             pad_h - pad_h // 2
                         ],
                                          dtype='int32').reshape([-1]),
                         data_format='NHWC')
        return x

    def depad_if_needed(self, x, size):
        n, h, w, c = size
        pad_h = paddle.cast(
            paddle.ceil(h / self.lgs[0]) * self.lgs[0] - h,
            paddle.int32).reshape([-1])
        pad_w = paddle.cast(
            paddle.ceil(w / self.lgs[1]) * self.lgs[1] - w,
            paddle.int32).reshape([-1])
        if pad_h > 0 or pad_w > 0:  # remove the center-padding on feature
            return x[:, pad_h // 2:pad_h // 2 + h, pad_w // 2:pad_w // 2 + w, :]
        return x


class LocalPermuteHelper:
    """ Permute the feature map to gather pixels in local groups, and then reverse permutation."""

    def __init__(self, local_group_size=7):
        self.lgs = local_group_size
        if not isinstance(self.lgs, (tuple, list)):
            self.lgs = to_2tuple(self.lgs)
        assert len(self.lgs) == 2, "The length of self.lgs must be 2."

    def permute(self, x, size):
        n, h, w, c = size
        qh = h // self.lgs[0]
        ph = self.lgs[0]
        qw = w // self.lgs[0]
        pw = self.lgs[0]
        c = c

        x = x.reshape([n, qh, ph, qw, pw, c])
        x = x.transpose([2, 4, 0, 1, 3, 5])
        x = x.reshape([ph * pw, n * qh * qw, c])
        return x

    def rev_permute(self, x, size):
        n, h, w, c = size
        x = x.reshape([
            self.lgs[0], self.lgs[0], n, h // self.lgs[0], w // self.lgs[0], c
        ])
        x = x.transpose([2, 3, 0, 4, 1, 5])
        x = x.reshape([n, h, w, c])

        return x


class Attention(nn.MultiHeadAttention):
    """ Multihead Attention with extra flags on the q/k/v and out projections."""

    def __init__(self,
                 *args,
                 add_zero_attn=None,
                 rpe=False,
                 window_size=7,
                 **kwargs):
        super(Attention, self).__init__(*args, **kwargs)

        self.add_zero_attn = add_zero_attn
        self.rpe = rpe
        if rpe:
            self.window_size = [window_size] * 2
            # define a parameter table of relative position bias
            parameter_value = paddle.zeros([
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                self.num_heads
            ])
            self.relative_position_bias_table = self.create_parameter(
                shape=parameter_value.shape,
                dtype=str(parameter_value.numpy().dtype),
                default_initializer=nn.initializer.Assign(parameter_value))

            # get pair-wise relative position index for each token inside the window
            coords_h = paddle.arange(self.window_size[0])
            coords_w = paddle.arange(self.window_size[1])
            coords = paddle.stack(paddle.meshgrid([coords_h,
                                                   coords_w]))  # 2, Wh, Ww
            coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = (coords_flatten[:, :, None] -
                               coords_flatten[:, None, :])  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.transpose([1, 2,
                                                         0])  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[
                0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index",
                                 relative_position_index)
            trunc_normal_(self.relative_position_bias_table)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
        do_qkv_proj=True,
        do_out_proj=True,
        rpe=True,
    ):

        tgt_len, bsz, embed_dim = query.shape
        head_dim = embed_dim // self.num_heads
        v_head_dim = self.vdim // self.num_heads
        assert (head_dim * self.num_heads == embed_dim
                ), "embed_dim must be divisible by num_heads"
        scaling = float(head_dim)**-0.5

        # whether or not use the original query/key/value
        q = self.q_proj(query) * scaling if do_qkv_proj else query
        k = self.k_proj(key) if do_qkv_proj else key
        v = self.v_proj(value) if do_qkv_proj else value

        if attn_mask is not None:
            dtype_lst = [
                paddle.float32, paddle.float64, paddle.float16, paddle.uint8,
                paddle.bool
            ]
            assert attn_mask.dtype in dtype_lst, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
            if attn_mask.dtype == paddle.uint8:
                msg = "Byte tensor for attn_mask in nn.MultiHeadAttention is deprecated. Use bool tensor instead."
                logger.warning(msg)
                attn_mask = attn_mask.to(paddle.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.shape) != [1, query.shape[0], key.shape[0]]:
                    raise RuntimeError(
                        "The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if attn_mask.shape != [
                        bsz * self.num_heads, query.shape[0], key.shape[0]
                ]:

                    raise RuntimeError(
                        "The size of the 3D attn_mask is not correct.")
            else:
                raise RuntimeError(
                    f"attn_mask's dimension {attn_mask.dim()} is not supported")

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == paddle.uint8:
            msg = "Byte tensor for key_padding_mask in nn.MultiHeadAttention is deprecated. Use bool tensor instead."
            logger.warning(msg)
            key_padding_mask = key_padding_mask.to(paddle.bool)

        q = q.reshape([tgt_len, bsz * self.num_heads,
                       head_dim]).transpose([1, 0, 2])
        if k is not None:
            k = k.reshape([-1, bsz * self.num_heads,
                           head_dim]).transpose([1, 0, 2])
        if v is not None:
            v = v.reshape([-1, bsz * self.num_heads,
                           v_head_dim]).transpose([1, 0, 2])

        src_len = k.shape[1]

        if key_padding_mask is not None:
            assert key_padding_mask.shape[0] == bsz
            assert key_padding_mask.shape[1] == src_len

        if self.add_zero_attn:
            src_len += 1
            k = paddle.concat(
                [k,
                 paddle.zeros((k.shape[0], 1) + k.shape[2:], dtype=k.dtype)],
                axis=1)
            v = paddle.concat(
                [v,
                 paddle.zeros((v.shape[0], 1) + v.shape[2:], dtype=v.dtype)],
                axis=1)

            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        attn_output_weights = paddle.bmm(q, k.transpose([0, 2, 1]))
        assert list(attn_output_weights.shape) == [
            bsz * self.num_heads, tgt_len, src_len
        ]
        """ Add relative position embedding."""
        if self.rpe and rpe:
            # NOTE: for simplicity, we assume src_len == tgt_len == window_size**2 here
            assert (
                src_len == self.window_size[0] * self.window_size[1]
                and tgt_len == self.window_size[0] * self.window_size[1]
            ), f"src{src_len}, tgt{tgt_len}, window{self.window_size[0]}"
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.reshape([-1])].reshape([
                    self.window_size[0] * self.window_size[1],
                    self.window_size[0] * self.window_size[1], -1
                ])  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.transpose(
                [2, 0, 1])  # nH, Wh*Ww, Wh*Ww
            attn_output_weights = attn_output_weights.reshape([
                bsz, self.num_heads, tgt_len, src_len
            ]) + relative_position_bias.unsqueeze(0)
            attn_output_weights = attn_output_weights.reshape(
                [bsz * self.num_heads, tgt_len, src_len])
        # Attention weight for the invalid region is -inf.
        if attn_mask is not None:
            if attn_mask.dtype == paddle.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.reshape(
                [bsz, self.num_heads, tgt_len, src_len])
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.reshape(
                [bsz * self.num_heads, tgt_len, src_len])

        attn_output_weights = F.softmax(attn_output_weights, axis=-1)
        attn_output_weights = F.dropout(attn_output_weights,
                                        p=self.dropout,
                                        training=self.training)

        attn_output = paddle.bmm(attn_output_weights, v)
        assert list(
            attn_output.shape) == [bsz * self.num_heads, tgt_len, v_head_dim]
        attn_output = (attn_output.transpose([1, 0, 2]).reshape(
            [tgt_len, bsz, self.vdim]))
        if do_out_proj:
            attn_output = F.linear(attn_output, self.out_proj.weight,
                                   self.out_proj.bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.reshape(
                [bsz, self.num_heads, tgt_len, src_len])
            return attn_output, q, k, attn_output_weights.sum(
                axis=1) / self.num_heads
        else:
            return attn_output, q, k  # additionaly return the query and key


class InterlacedPoolAttention(nn.Layer):
    """ Interlaced sparse multi-head self attention module with relative position bias.
    Args:
        embed_dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int, optional): Window size. Default: 7.
        rpe (bool, optional): Whether to use rpe. Default: True.
    """

    def __init__(self, embed_dim, num_heads, window_size=7, rpe=True, **kwargs):
        super(InterlacedPoolAttention, self).__init__()

        self.dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.with_rpe = rpe
        self.attn = Attention(embed_dim,
                              num_heads,
                              rpe=rpe,
                              window_size=window_size,
                              **kwargs)
        self.pad_helper = PadHelper(window_size)
        self.permute_helper = LocalPermuteHelper(window_size)

    def forward(self, x, H, W, **kwargs):
        B, N, C = x.shape
        x = x.reshape([B, H, W, C])
        # pad
        x_pad = self.pad_helper.pad_if_needed(x, x.shape)
        # permute
        x_permute = self.permute_helper.permute(x_pad, x_pad.shape)
        # attention
        out, _, _ = self.attn(x_permute,
                              x_permute,
                              x_permute,
                              rpe=self.with_rpe,
                              **kwargs)
        # reverse permutation
        out = self.permute_helper.rev_permute(out, x_pad.shape)
        out = self.pad_helper.depad_if_needed(out, x.shape)
        return out.reshape([B, N, C])


class MlpDWBN(nn.Layer):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 dw_act_layer=nn.GELU):
        super(MlpDWBN, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.act1 = act_layer()
        self.fc1 = layers.ConvBN(in_features, hidden_features, kernel_size=1)
        self.act2 = dw_act_layer()
        self.dw3x3 = layers.ConvBN(hidden_features,
                                   hidden_features,
                                   kernel_size=3,
                                   stride=1,
                                   groups=hidden_features,
                                   padding=1)

        self.fc2 = layers.ConvBN(hidden_features, out_features, kernel_size=1)
        self.act3 = act_layer()

    def forward(self, x, H, W):
        if x.dim() == 3:
            B, N, C = x.shape
            if N == (H * W + 1):
                cls_tokens = x[:, 0, :]
                x_ = x[:, 1:, :].transpose([0, 2, 1]).reshape([B, C, H, W])
            else:
                x_ = x.transpose([0, 2, 1]).reshape([B, C, H, W])

            x_ = self.fc1(x_)
            x_ = self.act1(x_)
            x_ = self.dw3x3(x_)
            x_ = self.act2(x_)
            x_ = self.fc2(x_)
            x_ = self.act3(x_)
            x_ = x_.reshape([B, C, -1]).transpose([0, 2, 1])
            if N == (H * W + 1):
                x = paddle.concat((cls_tokens.unsqueeze(1), x_), axis=1)
            else:
                x = x_
            return x

        elif x.dim() == 4:
            x = self.fc1(x)
            x = self.act1(x)
            x = self.dw3x3(x)
            x = self.act2(x)
            x = self.fc2(x)
            x = self.act3(x)
            return x

        else:
            raise RuntimeError(f"Unsupported input shape: {x.shape}")


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = layers.ConvBNReLU(inplanes,
                                       planes,
                                       kernel_size=1,
                                       bias_attr=False)
        self.conv2 = layers.ConvBNReLU(planes,
                                       planes,
                                       kernel_size=3,
                                       stride=stride,
                                       padding=1,
                                       bias_attr=False)
        self.conv3 = layers.ConvBN(planes,
                                   planes * self.expansion,
                                   kernel_size=1,
                                   bias_attr=False)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class GeneralTransformerBlock(nn.Layer):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.0,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(GeneralTransformerBlock, self).__init__()
        self.dim = inplanes
        self.out_dim = planes
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.attn = InterlacedPoolAttention(
            self.dim,
            num_heads=num_heads,
            window_size=window_size,
            rpe=True,
            dropout=attn_drop,
        )

        self.norm1 = norm_layer(self.dim, epsilon=1e-6)
        self.norm2 = norm_layer(self.dim, epsilon=1e-6)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(self.dim * mlp_ratio)

        self.mlp = MlpDWBN(in_features=self.dim,
                           hidden_features=mlp_hidden_dim,
                           out_features=self.out_dim,
                           act_layer=act_layer,
                           dw_act_layer=act_layer)

    def forward(self, x):
        B, C, H, W = x.shape
        # reshape
        x = x.reshape([B, C, -1]).transpose([0, 2, 1])
        # Attention
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        # reshape
        x = x.transpose([0, 2, 1]).reshape([B, C, H, W])
        return x


class HighResolutionTransformerModule(nn.Layer):

    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        num_inchannels,
        num_channels,
        num_heads,
        num_window_sizes,
        num_mlp_ratios,
        multi_scale_output=True,
        drop_path=0.0,
    ):
        super(HighResolutionTransformerModule, self).__init__()

        self._check_branches(num_branches, num_blocks, num_inchannels,
                             num_channels)

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(
            num_branches,
            blocks,
            num_blocks,
            num_channels,
            num_heads,
            num_window_sizes,
            num_mlp_ratios,
            drop_path,
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU()

        self.num_heads = num_heads
        self.num_window_sizes = num_window_sizes
        self.num_mlp_ratios = num_mlp_ratios

    def _check_branches(self, num_branches, num_blocks, num_inchannels,
                        num_channels):

        if num_branches != len(num_blocks):
            error_msg = f"Num_branches {num_branches} is not equal\
                to the length of num_blocks {len(num_blocks)}"

            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = f"Num_branches {num_branches} is not equal\
                to the length of num_channels {len(num_channels)}"

            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = f"Num_branches {num_branches} is not equal\
                to the length of num_inchannels {len(num_inchannels)}"

            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         num_heads, num_window_sizes, num_mlp_ratios,
                         drop_paths):

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                num_heads=num_heads[branch_index],
                window_size=num_window_sizes[branch_index],
                mlp_ratio=num_mlp_ratios[branch_index],
                drop_path=drop_paths[0],
            ))

        self.num_inchannels[
            branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                    num_heads=num_heads[branch_index],
                    window_size=num_window_sizes[branch_index],
                    mlp_ratio=num_mlp_ratios[branch_index],
                    drop_path=drop_paths[i],
                ))
        return nn.Sequential(*layers)

    def _make_branches(
        self,
        num_branches,
        block,
        num_blocks,
        num_channels,
        num_heads,
        num_window_sizes,
        num_mlp_ratios,
        drop_paths,
    ):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(
                    i,
                    block,
                    num_blocks,
                    num_channels,
                    num_heads,
                    num_window_sizes,
                    num_mlp_ratios,
                    drop_paths=[_ * (2**i) for _ in drop_paths]))

        return nn.LayerList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            layers.ConvBN(
                                num_inchannels[j],
                                num_inchannels[i],
                                kernel_size=1,
                                stride=1,
                                bias_attr=False,
                            ),
                            nn.Upsample(scale_factor=2**(j - i),
                                        mode="nearest")))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    layers.ConvBN(num_inchannels[j],
                                                  num_inchannels[j],
                                                  kernel_size=3,
                                                  stride=2,
                                                  padding=1,
                                                  groups=num_inchannels[j],
                                                  bias_attr=False),
                                    layers.ConvBN(num_inchannels[j],
                                                  num_outchannels_conv3x3,
                                                  kernel_size=1,
                                                  stride=1,
                                                  bias_attr=False)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    layers.ConvBN(num_inchannels[j],
                                                  num_inchannels[j],
                                                  kernel_size=3,
                                                  stride=2,
                                                  padding=1,
                                                  groups=num_inchannels[j],
                                                  bias_attr=False),
                                    layers.ConvBNReLU(num_inchannels[j],
                                                      num_outchannels_conv3x3,
                                                      kernel_size=1,
                                                      stride=1,
                                                      bias_attr=False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.LayerList(fuse_layer))

        return nn.LayerList(fuse_layers)

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode="bilinear",
                        align_corners=True,
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class HighResolutionTransformer(nn.Layer):
    """
    The HRFormer implementation based on PaddlePaddle.

    The original article refers to
    Jingdong Wang, et, al. "HRNet：Deep High-Resolution Representation Learning for Visual Recognition"
    (https://arxiv.org/pdf/1908.07919.pdf).

    Args:
        drop_path_rate (float, optional): The rate of Drop Path. Default: 0.2.
        stage1_num_blocks (list[int], optional): Number of blocks per module for stage1. Default: [2].
        stage1_num_channels (list[int], optional): Number of channels per branch for stage1. Default: [64].
        stage2_num_modules (int, optional): Number of modules for stage2. Default: 1.
        stage2_num_branches (int, optional): Number of branches for stage2. Default: 2.
        stage2_num_blocks (list[int], optional): Number of blocks per module for stage2. Default: [2, 2].
        stage2_num_channels (list[int], optional): Number of channels per branch for stage2. Default: [32, 64].
        stage2_num_heads (list[int], optional): Number of heads per multi head attetion for stage2. Default: [1, 2].
        stage2_num_mlp_ratios (list[int], optional): Number of ratio of mlp per multi head attetion for stage2. Default: [4, 4].
        stage2_num_window_sizes (list[int], optional): Number of window sizes for stage2. Default: [7, 7].
        stage3_num_modules (int, optional): Number of modules for stage3. Default: 4.
        stage3_num_branches (int, optional): Number of branches for stage3. Default: 3.
        stage3_num_blocks (list[int], optional): Number of blocks per module for stage3. Default: [2, 2, 2].
        stage3_num_channels (list[int], optional): Number of channels per branch for stage3. Default: [32, 64, 128].
        stage3_num_heads (list[int], optional): Number of heads per multi head attetion for stage3. Default: [1, 2, 4].
        stage3_num_mlp_ratios (list[int], optional): Number of ratio of mlp per multi head attetion for stage3. Default: [4, 4, 4].
        stage3_num_window_sizes (list[int], optional): Number of window sizes for stage3. Default: [7, 7, 7].
        stage4_num_modules (int, optional): Number of modules for stage4. Default: 2.
        stage4_num_branches (int, optional): Number of branches for stage4. Default: 4.
        stage4_num_blocks (list[int], optional): Number of blocks per module for stage4. Default: [2, 2, 2, 2].
        stage4_num_channels (list[int], optional): Number of channels per branch for stage4. Default: [32, 64, 128, 256].
        stage4_num_heads (list[int], optional): Number of heads per multi head attetion for stage4. Default: [1, 2, 4, 8].
        stage4_num_mlp_ratios (list[int], optional): Number of ratio of mlp per multi head attetion for stage4. Default: [4, 4, 4, 4].
        stage4_num_window_sizes (list[int], optional): Number of window sizes for stage4. Default: [7, 7, 7, 7].
        in_channels (int, optional): The channels of input image. Default: 3.
        pretrained (str, optional): The path of pretrained model. Default: None.
    """

    def __init__(self,
                 drop_path_rate=0.2,
                 stage1_num_blocks=[2],
                 stage1_num_channels=[64],
                 stage2_num_modules=1,
                 stage2_num_branches=2,
                 stage2_num_blocks=[2, 2],
                 stage2_num_channels=[32, 64],
                 stage2_num_heads=[1, 2],
                 stage2_num_mlp_ratios=[4, 4],
                 stage2_num_window_sizes=[7, 7],
                 stage3_num_modules=4,
                 stage3_num_branches=3,
                 stage3_num_blocks=[2, 2, 2],
                 stage3_num_channels=[32, 64, 128],
                 stage3_num_heads=[1, 2, 4],
                 stage3_num_mlp_ratios=[4, 4, 4],
                 stage3_num_window_sizes=[7, 7, 7],
                 stage4_num_modules=2,
                 stage4_num_branches=4,
                 stage4_num_blocks=[2, 2, 2, 2],
                 stage4_num_channels=[32, 64, 128, 256],
                 stage4_num_heads=[1, 2, 4, 8],
                 stage4_num_mlp_ratios=[4, 4, 4, 4],
                 stage4_num_window_sizes=[7, 7, 7, 7],
                 in_channels=3,
                 pretrained=None):
        super(HighResolutionTransformer, self).__init__()

        self.pretrained = pretrained
        self.drop_path_rate = drop_path_rate
        self.stage1_num_blocks = stage1_num_blocks
        self.stage1_num_channels = stage1_num_channels
        self.stage2_num_modules = stage2_num_modules
        self.stage2_num_branches = stage2_num_branches
        self.stage2_num_blocks = stage2_num_blocks
        self.stage2_num_channels = stage2_num_channels
        self.stage2_num_heads = stage2_num_heads
        self.stage2_num_mlp_ratios = stage2_num_mlp_ratios
        self.stage2_num_window_sizes = stage2_num_window_sizes
        self.stage3_num_modules = stage3_num_modules
        self.stage3_num_branches = stage3_num_branches
        self.stage3_num_blocks = stage3_num_blocks
        self.stage3_num_channels = stage3_num_channels
        self.stage3_num_heads = stage3_num_heads
        self.stage3_num_mlp_ratios = stage3_num_mlp_ratios
        self.stage3_num_window_sizes = stage3_num_window_sizes
        self.stage4_num_modules = stage4_num_modules
        self.stage4_num_branches = stage4_num_branches
        self.stage4_num_blocks = stage4_num_blocks
        self.stage4_num_channels = stage4_num_channels
        self.stage4_num_heads = stage4_num_heads
        self.stage4_num_mlp_ratios = stage4_num_mlp_ratios
        self.stage4_num_window_sizes = stage4_num_window_sizes

        self.conv1 = layers.ConvBNReLU(in_channels,
                                       64,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       bias_attr=False)

        self.conv2 = layers.ConvBNReLU(64,
                                       64,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       bias_attr=False)

        self.feat_channels = [sum(self.stage4_num_channels)]

        depth_s2 = self.stage2_num_blocks[0] * self.stage2_num_modules
        depth_s3 = self.stage3_num_blocks[0] * self.stage3_num_modules
        depth_s4 = self.stage4_num_blocks[0] * self.stage4_num_modules
        depths = [depth_s2, depth_s3, depth_s4]
        drop_path_rate = self.drop_path_rate

        dpr = [
            x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))
        ]

        num_channels = self.stage1_num_channels[0]
        block = Bottleneck
        num_blocks = self.stage1_num_blocks[0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        num_channels = self.stage2_num_channels
        block = GeneralTransformerBlock
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([stage1_out_channel],
                                                       num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            block=block,
            num_modules=self.stage2_num_modules,
            num_branches=self.stage2_num_branches,
            num_blocks=self.stage2_num_blocks,
            num_channels=self.stage2_num_channels,
            num_heads=self.stage2_num_heads,
            num_window_sizes=self.stage2_num_window_sizes,
            num_mlp_ratios=self.stage2_num_mlp_ratios,
            num_inchannels=num_channels,
            drop_path=dpr[0:depth_s2])

        num_channels = self.stage3_num_channels
        block = GeneralTransformerBlock
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            block=block,
            num_modules=self.stage3_num_modules,
            num_branches=self.stage3_num_branches,
            num_blocks=self.stage3_num_blocks,
            num_channels=self.stage3_num_channels,
            num_heads=self.stage3_num_heads,
            num_window_sizes=self.stage3_num_window_sizes,
            num_mlp_ratios=self.stage3_num_mlp_ratios,
            num_inchannels=num_channels,
            drop_path=dpr[depth_s2:depth_s2 + depth_s3])

        num_channels = self.stage4_num_channels
        block = GeneralTransformerBlock
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            block=block,
            num_modules=self.stage4_num_modules,
            num_branches=self.stage4_num_branches,
            num_blocks=self.stage4_num_blocks,
            num_channels=self.stage4_num_channels,
            num_heads=self.stage4_num_heads,
            num_window_sizes=self.stage4_num_window_sizes,
            num_mlp_ratios=self.stage4_num_mlp_ratios,
            num_inchannels=num_channels,
            multi_scale_output=True,
            drop_path=dpr[depth_s2 + depth_s3:])

        self.init_weight()

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            layers.ConvBNReLU(num_channels_pre_layer[i],
                                              num_channels_cur_layer[i],
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              bias_attr=False)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (num_channels_cur_layer[i] if j == i -
                                   num_branches_pre else inchannels)
                    conv3x3s.append(
                        nn.Sequential(
                            layers.ConvBNReLU(inchannels,
                                              outchannels,
                                              kernel_size=3,
                                              stride=2,
                                              padding=1,
                                              bias_attr=False)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.LayerList(transition_layers)

    def _make_layer(self,
                    block,
                    inplanes,
                    planes,
                    blocks,
                    num_heads=1,
                    stride=1,
                    window_size=7,
                    mlp_ratio=4.0):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                layers.ConvBN(inplanes,
                              planes * block.expansion,
                              kernel_size=1,
                              stride=stride,
                              bias_attr=False))
        modules = []

        if isinstance(block, GeneralTransformerBlock):
            modules.append(
                block(
                    inplanes,
                    planes,
                    num_heads,
                    window_size,
                    mlp_ratio,
                ))
        else:
            modules.append(block(inplanes, planes, stride, downsample))

        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            modules.append(block(inplanes, planes))

        return nn.Sequential(*modules)

    def _make_stage(self,
                    block,
                    num_modules,
                    num_branches,
                    num_blocks,
                    num_channels,
                    num_heads,
                    num_window_sizes,
                    num_mlp_ratios,
                    num_inchannels,
                    multi_scale_output=True,
                    drop_path=0.0):
        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionTransformerModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    num_heads,
                    num_window_sizes,
                    num_mlp_ratios,
                    reset_multi_scale_output,
                    drop_path=drop_path[num_blocks[0] * i:num_blocks[0] *
                                        (i + 1)],
                ))
            num_inchannels = modules[-1].num_inchannels

        return nn.Sequential(*modules), num_inchannels

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)
            elif isinstance(layer, nn.Linear):
                trunc_normal_(layer.weight)
                if layer.bias is not None:
                    zeros_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                zeros_(layer.bias)
                ones_(layer.weight)

        if self.pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_num_branches):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_num_branches):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_num_branches):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        _, _, h, w = y_list[0].shape
        feat1 = y_list[0]
        feat2 = F.interpolate(y_list[1],
                              size=(h, w),
                              mode="bilinear",
                              align_corners=True)
        feat3 = F.interpolate(y_list[2],
                              size=(h, w),
                              mode="bilinear",
                              align_corners=True)
        feat4 = F.interpolate(y_list[3],
                              size=(h, w),
                              mode="bilinear",
                              align_corners=True)

        feats = paddle.concat([feat1, feat2, feat3, feat4], axis=1)

        return [feats]


@manager.BACKBONES.add_component
def HRFormer_small(**kwargs):
    arch_net = HighResolutionTransformer(**kwargs)

    return arch_net


@manager.BACKBONES.add_component
def HRFormer_base(**kwargs):
    arch_net = HighResolutionTransformer(
        stage2_num_channels=[78, 156],
        stage2_num_heads=[2, 4],
        stage3_num_channels=[78, 156, 312],
        stage3_num_heads=[2, 4, 8],
        stage4_num_channels=[78, 156, 312, 624],
        stage4_num_heads=[2, 4, 8, 16],
        **kwargs)

    return arch_net


@manager.BACKBONES.add_component
def HRFormer_base_win_13(**kwargs):
    arch_net = HighResolutionTransformer(
        stage2_num_channels=[78, 156],
        stage2_num_heads=[2, 4],
        stage2_num_window_sizes=[13, 13],
        stage3_num_channels=[78, 156, 312],
        stage3_num_heads=[2, 4, 8],
        stage4_num_channels=[78, 156, 312, 624],
        stage4_num_heads=[2, 4, 8, 16],
        **kwargs)

    return arch_net


@manager.BACKBONES.add_component
def HRFormer_base_win_15(**kwargs):
    arch_net = HighResolutionTransformer(
        stage2_num_channels=[78, 156],
        stage2_num_heads=[2, 4],
        stage2_num_window_sizes=[15, 15],
        stage3_num_channels=[78, 156, 312],
        stage3_num_heads=[2, 4, 8],
        stage3_num_window_sizes=[15, 15, 15],
        stage4_num_channels=[78, 156, 312, 624],
        stage4_num_heads=[2, 4, 8, 16],
        stage4_num_window_sizes=[15, 15, 15, 15],
        **kwargs)

    return arch_net

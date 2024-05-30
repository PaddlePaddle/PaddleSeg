# This file is heavily based on https://github.com/czczup/ViT-Adapter
import math
import warnings

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import param_init
from paddleseg.cvlibs.param_init import constant_init, xavier_uniform


class MSDeformAttn(nn.Layer):

    def __init__(self,
                 d_model=256,
                 n_levels=4,
                 n_heads=8,
                 n_points=4,
                 ratio=1.0):
        """Multi-Scale Deformable Attention Module.
        
        Args:
            d_model(int, optional): The hidden dimension. Default: 256
            n_levels(int, optional): The number of feature levels. Default: 4
            n_heads(int, optional): The number of attention heads. Default: 8
            n_points(int, optional): The number of sampling points per attention head per feature level. Default: 4
            ratio (float, optional): The ratio of channels for Linear. Default: 1.0
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, '
                             'but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2
        # which is more efficient in our CUDA implementation
        if not self._is_power_of_2(_d_per_head):
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

    @staticmethod
    def _is_power_of_2(n):
        if (not isinstance(n, int)) or (n < 0):
            raise ValueError(
                'invalid input for _is_power_of_2: {} (type: {})'.format(
                    n, type(n)))
        return (n & (n - 1) == 0) and n != 0

    def _reset_parameters(self):
        constant_init(self.sampling_offsets.weight, value=0.)
        thetas = paddle.arange(self.n_heads,
                               dtype='float32') * (2.0 * math.pi / self.n_heads)
        grid_init = paddle.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).reshape(
                         [self.n_heads, 1, 1,
                          2]).tile([1, self.n_levels, self.n_points, 1])
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        grid_init = grid_init.reshape([-1])
        self.sampling_offsets.bias = self.create_parameter(
            shape=grid_init.shape,
            default_initializer=paddle.nn.initializer.Assign(grid_init))
        self.sampling_offsets.bias.stop_gradient = True

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
        Args:
            query:                       (N, Length_{query}, C)
            reference_points:            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                            or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
            input_flatten:               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
            input_spatial_shapes:        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            input_level_start_index:     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
            input_padding_mask:          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        Returns:
            output                     (N, Length_{query}, C)
        """

        def masked_fill(x, mask, value):
            y = paddle.full(x.shape, value, x.dtype)
            return paddle.where(mask, y, x)

        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] *
                input_spatial_shapes[:, 1]).sum() == Len_in

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
                -1).astype('float32')
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'
                .format(reference_points.shape[-1]))
        try:
            import ms_deform_attn
        except:
            print(
                "Import ms_deform_attn failed. Please download the following file and refer to "
                "the readme to install ms_deform_attn lib: "
                "https://paddleseg.bj.bcebos.com/dygraph/customized_ops/ms_deform_attn.zip"
            )
            exit()
        output = ms_deform_attn.ms_deform_attn(value, input_spatial_shapes,
                                               input_level_start_index,
                                               sampling_locations,
                                               attention_weights,
                                               self.im2col_step)
        output = self.output_proj(output)
        return output

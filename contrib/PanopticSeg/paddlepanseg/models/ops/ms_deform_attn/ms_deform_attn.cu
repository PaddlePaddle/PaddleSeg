/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// Adapted from https://github.com/facebookresearch/Mask2Former
// 
// Original copyright info: 

/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

/*!
* Copyright (c) Facebook, Inc. and its affiliates.
* Modified by Bowen Cheng from https://github.com/fundamentalvision/Deformable-DETR
*/

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <paddle/extension.h>

#include "ms_deform_im2col.cuh"


std::vector<paddle::Tensor> MSDeformAttnCUDAForward(
    const paddle::Tensor &value, 
    const paddle::Tensor &spatial_shapes,
    const paddle::Tensor &level_start_index,
    const paddle::Tensor &sampling_loc,
    const paddle::Tensor &attn_weight,
    const int im2col_step)
{
    PD_CHECK(value.is_gpu(), "`value` must be a CUDA tensor.");
    PD_CHECK(spatial_shapes.is_gpu(), "`spatial_shapes` must be a CUDA tensor.");
    PD_CHECK(level_start_index.is_gpu(), "`level_start_index` must be a CUDA tensor.");
    PD_CHECK(sampling_loc.is_gpu(), "`sampling_loc` must be a CUDA tensor.");
    PD_CHECK(attn_weight.is_gpu(), "`attn_weight` must be a CUDA tensor.");

    auto value_dims = value.shape();
    const int batch = value_dims[0];
    const int spatial_size = value_dims[1];
    const int num_heads = value_dims[2];
    const int channels = value_dims[3];

    const int num_levels = spatial_shapes.shape()[0];

    const int num_query = sampling_loc.shape()[1];
    const int num_point = sampling_loc.shape()[4];

    const int im2col_step_ = std::min(batch, im2col_step);

    PD_CHECK(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d).", batch, im2col_step_);
    
    auto output = paddle::full({batch, num_query, num_heads, channels}, 0, value.dtype(), value.place());

    const int batch_n = im2col_step_;
    auto output_n = paddle::reshape(output, {batch/im2col_step_, batch_n, num_query, num_heads, channels});
    auto per_value_size = spatial_size * num_heads * channels;
    auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
    auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
    for (int n = 0; n < batch/im2col_step_; ++n)
    {
        auto columns = output_n.slice(n, n+1);
        PD_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_cuda_forward", ([&] {
            ms_deformable_im2col_cuda(value.stream(),
                value.data<data_t>() + n * im2col_step_ * per_value_size,
                spatial_shapes.data<int64_t>(),
                level_start_index.data<int64_t>(),
                sampling_loc.data<data_t>() + n * im2col_step_ * per_sample_loc_size,
                attn_weight.data<data_t>() + n * im2col_step_ * per_attn_weight_size,
                batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                columns.data<data_t>());

        }));
    }

    output.reshape({batch, num_query, num_heads*channels});

    return {output};
}


std::vector<paddle::Tensor> MSDeformAttnCUDABackward(
    const paddle::Tensor &value, 
    const paddle::Tensor &spatial_shapes,
    const paddle::Tensor &level_start_index,
    const paddle::Tensor &sampling_loc,
    const paddle::Tensor &attn_weight,
    const paddle::Tensor &grad_output,
    const int im2col_step)
{
    PD_CHECK(value.is_gpu(), "`value` must be a CUDA tensor.");
    PD_CHECK(spatial_shapes.is_gpu(), "`spatial_shapes` must be a CUDA tensor.");
    PD_CHECK(level_start_index.is_gpu(), "`level_start_index` must be a CUDA tensor.");
    PD_CHECK(sampling_loc.is_gpu(), "`sampling_loc` must be a CUDA tensor.");
    PD_CHECK(attn_weight.is_gpu(), "`attn_weight` must be a CUDA tensor.");
    PD_CHECK(grad_output.is_gpu(), "`grad_output` must be a CUDA tensor.");

    auto value_dims = value.shape();
    const int batch = value_dims[0];
    const int spatial_size = value_dims[1];
    const int num_heads = value_dims[2];
    const int channels = value_dims[3];

    const int num_levels = spatial_shapes.shape()[0];

    const int num_query = sampling_loc.shape()[1];
    const int num_point = sampling_loc.shape()[4];

    const int im2col_step_ = std::min(batch, im2col_step);

    PD_CHECK(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);

    auto grad_value = paddle::full(value.shape(), 0, value.dtype(), value.place());
    auto grad_sampling_loc = paddle::full(sampling_loc.shape(), 0, sampling_loc.dtype(), sampling_loc.place());
    auto grad_attn_weight = paddle::full(attn_weight.shape(), 0, attn_weight.dtype(), attn_weight.place());

    const int batch_n = im2col_step_;
    auto per_value_size = spatial_size * num_heads * channels;
    auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
    auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
    auto grad_output_n = paddle::reshape(grad_output, {batch/im2col_step_, batch_n, num_query, num_heads, channels});
    
    for (int n = 0; n < batch/im2col_step_; ++n)
    {
        auto grad_output_g = grad_output_n.slice(n, n+1);
        PD_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_backward_cuda", ([&] {
            ms_deformable_col2im_cuda(value.stream(),
                                    grad_output_g.data<data_t>(),
                                    value.data<data_t>() + n * im2col_step_ * per_value_size,
                                    spatial_shapes.data<int64_t>(),
                                    level_start_index.data<int64_t>(),
                                    sampling_loc.data<data_t>() + n * im2col_step_ * per_sample_loc_size,
                                    attn_weight.data<data_t>() + n * im2col_step_ * per_attn_weight_size,
                                    batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                                    grad_value.data<data_t>() +  n * im2col_step_ * per_value_size,
                                    grad_sampling_loc.data<data_t>() + n * im2col_step_ * per_sample_loc_size,
                                    grad_attn_weight.data<data_t>() + n * im2col_step_ * per_attn_weight_size);

        }));
    }

    return {
        grad_value, grad_sampling_loc, grad_attn_weight
    };
}

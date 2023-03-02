/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <vector>

#include <paddle/extension.h>


std::vector<paddle::Tensor> MSDeformAttnCPUForward(
    const paddle::Tensor &value, 
    const paddle::Tensor &spatial_shapes,
    const paddle::Tensor &level_start_index,
    const paddle::Tensor &sampling_loc,
    const paddle::Tensor &attn_weight,
    const int im2col_step) {
    PD_THROW("Not Implemented on CPU.");
}


std::vector<paddle::Tensor> MSDeformAttnCPUBackward(
    const paddle::Tensor &value, 
    const paddle::Tensor &spatial_shapes,
    const paddle::Tensor &level_start_index,
    const paddle::Tensor &sampling_loc,
    const paddle::Tensor &attn_weight,
    const paddle::Tensor &grad_output,
    const int im2col_step) {
    PD_THROW("Not Implemented on CPU.");
}


std::vector<paddle::Tensor> MSDeformAttnCUDAForward(
    const paddle::Tensor &value, 
    const paddle::Tensor &spatial_shapes,
    const paddle::Tensor &level_start_index,
    const paddle::Tensor &sampling_loc,
    const paddle::Tensor &attn_weight,
    const int im2col_step);
    

std::vector<paddle::Tensor> MSDeformAttnCUDABackward(
    const paddle::Tensor &value, 
    const paddle::Tensor &spatial_shapes,
    const paddle::Tensor &level_start_index,
    const paddle::Tensor &sampling_loc,
    const paddle::Tensor &attn_weight,
    const paddle::Tensor &grad_output,
    const int im2col_step);


std::vector<paddle::Tensor> MSDeformAttnForward(
    const paddle::Tensor &value, 
    const paddle::Tensor &spatial_shapes,
    const paddle::Tensor &level_start_index,
    const paddle::Tensor &sampling_loc,
    const paddle::Tensor &attn_weight,
    const int im2col_step) {
    if (value.is_gpu()) {
        return MSDeformAttnCUDAForward(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step);
    } else if (value.is_cpu()) {
        return MSDeformAttnCPUForward(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step);
    } else {
        PD_THROW("Unsupported device type.");
    }
}
    

std::vector<paddle::Tensor> MSDeformAttnBackward(
    const paddle::Tensor &value, 
    const paddle::Tensor &spatial_shapes,
    const paddle::Tensor &level_start_index,
    const paddle::Tensor &sampling_loc,
    const paddle::Tensor &attn_weight,
    const paddle::Tensor &grad_output,
    const int im2col_step) {
    if (value.is_gpu()) {
        return MSDeformAttnCUDABackward(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output, im2col_step);
    } else if (value.is_cpu()) {
        return MSDeformAttnCPUBackward(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output, im2col_step);
    } else {
        PD_THROW("Unsupported device type.");
    }
}

std::vector<std::vector<int64_t>> InferShape(const std::vector<int64_t> value_shape,
                                             const std::vector<int64_t> spatial_shapes_shape,
                                             const std::vector<int64_t> level_start_index_shape,
                                             const std::vector<int64_t> sampling_loc_shape,
                                             const std::vector<int64_t> attn_weight_shape,
                                             const int im2col_step) {
    const int batch = value_shape[0];
    const int num_heads = value_shape[2];
    const int channels = value_shape[3];
    const int num_query = sampling_loc_shape[1];
    return {{batch, num_query, num_heads*channels}};
}

std::vector<paddle::DataType>
InferDtype(paddle::DataType value_dtype,
           paddle::DataType spatial_shapes_dtype,
           paddle::DataType level_start_index_dtype,
           paddle::DataType sampling_loc_dtype,
           paddle::DataType attn_weight_dtype) {
    return {value_dtype};
}


PD_BUILD_OP(ms_deform_attn)
    .Inputs({"value", "spatial_shapes", "level_start_index", "sampling_loc", "attn_weight"})
    .Outputs({"output"})
    .Attrs({"im2col_step: int"})
    .SetKernelFn(PD_KERNEL(MSDeformAttnForward))
    .SetInferShapeFn(PD_INFER_SHAPE(InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(InferDtype));

PD_BUILD_GRAD_OP(ms_deform_attn)
    .Inputs({"value", "spatial_shapes", "level_start_index", "sampling_loc", "attn_weight", paddle::Grad("output")})
    .Outputs({paddle::Grad("value"), paddle::Grad("sampling_loc"), paddle::Grad("attn_weight")})
    .Attrs({"im2col_step: int"})
    .SetKernelFn(PD_KERNEL(MSDeformAttnBackward));

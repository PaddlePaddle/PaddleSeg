# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle

import paddle.inference as paddle_infer


def load_model(model_path, param_path, use_gpu=None):
    config = paddle_infer.Config(model_path, param_path)
    if use_gpu is None:
        if paddle.device.is_compiled_with_cuda():  # TODO: 可以使用GPU却返回False
            use_gpu = True
        else:
            use_gpu = False
    if not use_gpu:
        config.enable_mkldnn()
        # TODO: fluid要废弃了，研究判断方式
        config.switch_ir_optim(True)
        config.set_cpu_math_library_num_threads(10)
    else:
        config.enable_use_gpu(500, 0)
        config.delete_pass("conv_elementwise_add_act_fuse_pass")
        config.delete_pass("conv_elementwise_add2_act_fuse_pass")
        config.delete_pass("conv_elementwise_add_fuse_pass")
        config.switch_ir_optim()
        config.enable_memory_optim()
    # config = paddle_infer.Config(model_path, param_path)
    # config.enable_mkldnn()
    # config.switch_ir_optim(True)
    # config.set_cpu_math_library_num_threads(10)
    model = paddle_infer.create_predictor(config)
    return model


def jit_load(path):
    model = paddle.jit.load(path)
    model.eval()
    return model


def calculate_memorize(model, frame, masks):
    input_names = model.get_input_names()
    frame_handle = model.get_input_handle(input_names[0])
    masks_handle = model.get_input_handle(input_names[1])
    frame_handle.copy_from_cpu(frame)
    masks_handle.copy_from_cpu(masks)
    model.run()
    output_names = model.get_output_names()
    output_handle = model.get_output_handle(output_names[0])
    result = output_handle.copy_to_cpu()
    output_handle2 = model.get_output_handle(output_names[1])
    result2 = output_handle2.copy_to_cpu()
    return result, result2


def calculate_segmentation(model, frame, keys, values):
    input_names = model.get_input_names()
    frame_handle = model.get_input_handle(input_names[0])
    keys_handle = model.get_input_handle(input_names[1])
    values_handle = model.get_input_handle(input_names[2])
    frame_handle.copy_from_cpu(frame)
    keys_handle.copy_from_cpu(keys)
    values_handle.copy_from_cpu(values)
    model.run()
    output_names = model.get_output_names()
    output_handle = model.get_output_handle(output_names[0])
    result = output_handle.copy_to_cpu()
    output_handle2 = model.get_output_handle(output_names[1])
    result2 = output_handle2.copy_to_cpu()
    return result, result2


def calculate_attention(model, mk16, qk16, pos_mask, neg_mask):
    input_names = model.get_input_names()
    mk16_handle = model.get_input_handle(input_names[0])
    qk16_handle = model.get_input_handle(input_names[1])
    pos_handle = model.get_input_handle(input_names[2])
    neg_handle = model.get_input_handle(input_names[3])
    mk16_handle.copy_from_cpu(mk16)
    qk16_handle.copy_from_cpu(qk16)
    pos_handle.copy_from_cpu(pos_mask)
    neg_handle.copy_from_cpu(neg_mask)
    model.run()
    output_names = model.get_output_names()
    output_handle = model.get_output_handle(output_names[0])
    result = output_handle.copy_to_cpu()
    return result


def calculate_fusion(model, im, seg1, seg2, attn, time):
    input_names = model.get_input_names()
    im_handle = model.get_input_handle(input_names[0])
    seg1_handle = model.get_input_handle(input_names[1])
    seg2_handle = model.get_input_handle(input_names[2])
    attn_handle = model.get_input_handle(input_names[3])
    time_handle = model.get_input_handle(input_names[4])
    im_handle.copy_from_cpu(im)
    seg1_handle.copy_from_cpu(seg1)
    seg2_handle.copy_from_cpu(seg2)
    attn_handle.copy_from_cpu(attn)
    time_handle.copy_from_cpu(time)
    model.run()
    output_names = model.get_output_names()
    output_handle = model.get_output_handle(output_names[0])
    result = output_handle.copy_to_cpu()
    return result

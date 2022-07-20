# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os.path as osp
from abc import abstractmethod

import paddle.inference as paddle_infer

here = osp.dirname(osp.abspath(__file__))


class EISegModel:
    @abstractmethod
    def __init__(self, model_path, param_path, use_gpu=False):
        model_path, param_path = self.check_param(model_path, param_path)
        try:
            config = paddle_infer.Config(model_path, param_path)
        except:
            ValueError(" 模型和参数不匹配，请检查模型和参数是否加载错误")
        if not use_gpu:
            config.enable_mkldnn()
            # TODO: fluid要废弃了，研究判断方式
            # if paddle.fluid.core.supports_bfloat16():
            #     config.enable_mkldnn_bfloat16()
            config.switch_ir_optim(True)
            config.set_cpu_math_library_num_threads(10)
        else:
            config.enable_use_gpu(500, 0)
            config.delete_pass("conv_elementwise_add_act_fuse_pass")
            config.delete_pass("conv_elementwise_add2_act_fuse_pass")
            config.delete_pass("conv_elementwise_add_fuse_pass")
            config.switch_ir_optim()
            config.enable_memory_optim()
            # use_tensoret = False  # TODO: 目前Linux和windows下使用TensorRT报错
            # if use_tensoret:
            #     config.enable_tensorrt_engine(
            #         workspace_size=1 << 30,
            #         precision_mode=paddle_infer.PrecisionType.Float32,
            #         max_batch_size=1,
            #         min_subgraph_size=5,
            #         use_static=False,
            #         use_calib_mode=False,
            #     )
        self.model = paddle_infer.create_predictor(config)

    def check_param(self, model_path, param_path):
        if model_path is None or not osp.exists(model_path):
            raise Exception(f"模型路径{model_path}不存在。请指定正确的模型路径")
        if param_path is None or not osp.exists(param_path):
            raise Exception(f"权重路径{param_path}不存在。请指定正确的权重路径")
        return model_path, param_path

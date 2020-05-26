# coding: utf8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from utils.util import AttrDict, merge_cfg_from_args, get_arguments
import os

args = get_arguments()
cfg = AttrDict()

# 待预测图像所在路径
cfg.data_dir = os.path.join(args.example, "data", "test_images")
# 待预测图像名称列表
cfg.data_list_file = os.path.join(args.example, "data", "test.txt")
# 模型加载路径
cfg.model_path = os.path.join(args.example, "model")
# 预测结果保存路径
cfg.vis_dir = os.path.join(args.example, "result")

# 预测类别数
cfg.class_num = 2
# 均值, 图像预处理减去的均值
cfg.MEAN = 127.5, 127.5, 127.5
# 标准差，图像预处理除以标准差
cfg.STD = 127.5, 127.5, 127.5
# 待预测图像输入尺寸
cfg.input_size = 1536, 576

merge_cfg_from_args(args, cfg)

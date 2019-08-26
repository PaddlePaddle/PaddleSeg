# -*- coding: utf-8 -*-
from utils.util import AttrDict, get_arguments, merge_cfg_from_args
import os

args = get_arguments()
cfg = AttrDict()

# 待预测图像所在路径
cfg.data_dir = os.path.join(args.example , "data", "test_images")
# 待预测图像名称列表
cfg.data_list_file = os.path.join(args.example , "data", "test.txt")
# 模型加载路径
cfg.model_path = os.path.join(args.example , "model")
# 预测结果保存路径
cfg.vis_dir = os.path.join(args.example , "result")

# 预测类别数
cfg.class_num = 2
# 均值, 图像预处理减去的均值
cfg.MEAN = 104.008, 116.669, 122.675
# 标准差，图像预处理除以标准差
cfg.STD =  1.0, 1.0, 1.0
# 待预测图像输入尺寸
cfg.input_size = 513, 513

merge_cfg_from_args(args, cfg)

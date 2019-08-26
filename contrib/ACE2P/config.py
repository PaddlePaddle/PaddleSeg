# -*- coding: utf-8 -*-
from utils.util import AttrDict, merge_cfg_from_args, get_arguments
import os

args = get_arguments()
cfg = AttrDict()

# 待预测图像所在路径
cfg.data_dir = os.path.join(args.example , "data", "testing_images")
# 待预测图像名称列表
cfg.data_list_file = os.path.join(args.example , "data", "test_id.txt")
# 模型加载路径
cfg.model_path = os.path.join(args.example , "ACE2P")
# 预测结果保存路径
cfg.vis_dir = os.path.join(args.example , "result")

# 预测类别数
cfg.class_num = 20
# 均值, 图像预处理减去的均值
cfg.MEAN = 0.406, 0.456, 0.485
# 标准差，图像预处理除以标准差
cfg.STD =  0.225, 0.224, 0.229

# 多尺度预测时图像尺寸
cfg.multi_scales = (377,377), (473,473), (567,567)
# 多尺度预测时图像是否水平翻转
cfg.flip = True

merge_cfg_from_args(args, cfg)

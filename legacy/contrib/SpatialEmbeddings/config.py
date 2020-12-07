# -*- coding: utf-8 -*-
from utils.util import AttrDict, merge_cfg_from_args, get_arguments
import os

args = get_arguments()
cfg = AttrDict()

# 待预测图像所在路径
cfg.data_dir = "data"
# 待预测图像名称列表
cfg.data_list_file = os.path.join("data", "test.txt")
# 模型加载路径
cfg.model_path = 'SpatialEmbeddings_kitti'
# 预测结果保存路径
cfg.vis_dir = "result"

# 待预测图像输入尺寸
cfg.input_size = (384, 1248)
# sigma值
cfg.n_sigma = 2
# 中心点阈值
cfg.threshold = 0.94
# 点集数阈值
cfg.min_pixel = 160

merge_cfg_from_args(args, cfg)

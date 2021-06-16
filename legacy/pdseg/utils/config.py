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

from __future__ import print_function
from __future__ import unicode_literals
from utils.collect import SegConfig
import numpy as np

cfg = SegConfig()

########################## 基本配置 ###########################################
# 均值，图像预处理减去的均值
cfg.MEAN = [0.5, 0.5, 0.5]
# 标准差，图像预处理除以标准差·
cfg.STD = [0.5, 0.5, 0.5]
# 批处理大小
cfg.BATCH_SIZE = 1
# 验证时图像裁剪尺寸（宽，高）
cfg.EVAL_CROP_SIZE = tuple()
# 训练时图像裁剪尺寸（宽，高）
cfg.TRAIN_CROP_SIZE = tuple()
# 多进程训练总进程数
cfg.NUM_TRAINERS = 1
# 多进程训练进程ID
cfg.TRAINER_ID = 0
########################## 数据载入配置 #######################################
# 数据载入时的并发数, 建议值8
cfg.DATALOADER.NUM_WORKERS = 8
# 数据载入时缓存队列大小, 建议值256
cfg.DATALOADER.BUF_SIZE = 256

########################## 数据集配置 #########################################
# 数据主目录目录
cfg.DATASET.DATA_DIR = './dataset/cityscapes/'
# 训练集列表
cfg.DATASET.TRAIN_FILE_LIST = './dataset/cityscapes/train.list'
# 训练集数量
cfg.DATASET.TRAIN_TOTAL_IMAGES = 2975
# 验证集列表
cfg.DATASET.VAL_FILE_LIST = './dataset/cityscapes/val.list'
# 验证数据数量
cfg.DATASET.VAL_TOTAL_IMAGES = 500
# 测试数据列表
cfg.DATASET.TEST_FILE_LIST = './dataset/cityscapes/test.list'
# 测试数据数量
cfg.DATASET.TEST_TOTAL_IMAGES = 500
# VisualDL 可视化的数据集
cfg.DATASET.VIS_FILE_LIST = None
# 类别数(需包括背景类)
cfg.DATASET.NUM_CLASSES = 19
# 输入图像类型, 支持三通道'rgb',四通道'rgba',单通道灰度图'gray'
cfg.DATASET.IMAGE_TYPE = 'rgb'
# 输入图片的通道数
cfg.DATASET.DATA_DIM = 3
# 数据列表分割符, 默认为空格
cfg.DATASET.SEPARATOR = ' '
# 忽略的像素标签值, 默认为255，一般无需改动
cfg.DATASET.IGNORE_INDEX = 255
# 数据增强是图像的padding值
cfg.DATASET.PADDING_VALUE = [127.5, 127.5, 127.5]

########################### 数据增强配置 ######################################
# 图像resize的方式有三种：
# unpadding（固定尺寸），stepscaling（按比例resize），rangescaling（长边对齐）
cfg.AUG.AUG_METHOD = 'unpadding'
# 图像resize的固定尺寸（宽，高），非负
cfg.AUG.FIX_RESIZE_SIZE = (512, 512)
# 图像resize方式为stepscaling，resize最小尺度，非负
cfg.AUG.MIN_SCALE_FACTOR = 0.5
# 图像resize方式为stepscaling，resize最大尺度，不小于MIN_SCALE_FACTOR
cfg.AUG.MAX_SCALE_FACTOR = 2.0
# 图像resize方式为stepscaling，resize尺度范围间隔，非负
cfg.AUG.SCALE_STEP_SIZE = 0.25
# 图像resize方式为rangescaling，训练时长边resize的范围最小值，非负
cfg.AUG.MIN_RESIZE_VALUE = 400
# 图像resize方式为rangescaling，训练时长边resize的范围最大值，
# 不小于MIN_RESIZE_VALUE
cfg.AUG.MAX_RESIZE_VALUE = 600
# 图像resize方式为rangescaling, 测试验证可视化模式下长边resize的长度，
# 在MIN_RESIZE_VALUE到MAX_RESIZE_VALUE范围内
cfg.AUG.INF_RESIZE_VALUE = 500

# 图像镜像左右翻转
cfg.AUG.MIRROR = True
# 图像上下翻转开关，True/False
cfg.AUG.FLIP = False
# 图像启动上下翻转的概率，0-1
cfg.AUG.FLIP_RATIO = 0.5

# RichCrop数据增广开关，用于提升模型鲁棒性
cfg.AUG.RICH_CROP.ENABLE = False
# 图像旋转最大角度，0-90
cfg.AUG.RICH_CROP.MAX_ROTATION = 15
# 裁取图像与原始图像面积比，0-1
cfg.AUG.RICH_CROP.MIN_AREA_RATIO = 0.5
# 裁取图像宽高比范围，非负
cfg.AUG.RICH_CROP.ASPECT_RATIO = 0.33
# 亮度调节范围，0-1
cfg.AUG.RICH_CROP.BRIGHTNESS_JITTER_RATIO = 0.5
# 饱和度调节范围，0-1
cfg.AUG.RICH_CROP.SATURATION_JITTER_RATIO = 0.5
# 对比度调节范围，0-1
cfg.AUG.RICH_CROP.CONTRAST_JITTER_RATIO = 0.5
# 图像模糊开关，True/False
cfg.AUG.RICH_CROP.BLUR = False
# 图像启动模糊百分比，0-1
cfg.AUG.RICH_CROP.BLUR_RATIO = 0.1
# 图像是否切换到rgb模式
cfg.AUG.TO_RGB = False

########################### 训练配置 ##########################################
# 模型保存路径
cfg.TRAIN.MODEL_SAVE_DIR = ''
# 预训练模型路径
cfg.TRAIN.PRETRAINED_MODEL_DIR = ''
# 是否resume，继续训练
cfg.TRAIN.RESUME_MODEL_DIR = ''
# 是否使用多卡间同步BatchNorm均值和方差
cfg.TRAIN.SYNC_BATCH_NORM = False
# 模型参数保存的epoch间隔数，可用来继续训练中断的模型
cfg.TRAIN.SNAPSHOT_EPOCH = 10

########################### 模型优化相关配置 ##################################
# 初始学习率
cfg.SOLVER.LR = 0.1
# 学习率下降方法, 支持poly piecewise cosine 三种
cfg.SOLVER.LR_POLICY = "poly"
# 优化算法, 支持SGD和Adam两种算法
cfg.SOLVER.OPTIMIZER = "sgd"
# 动量参数
cfg.SOLVER.MOMENTUM = 0.9
# 二阶矩估计的指数衰减率
cfg.SOLVER.MOMENTUM2 = 0.999
# 学习率Poly下降指数
cfg.SOLVER.POWER = 0.9
# step下降指数
cfg.SOLVER.GAMMA = 0.1
# step下降间隔
cfg.SOLVER.DECAY_EPOCH = [10, 20]
# 学习率权重衰减，0-1
cfg.SOLVER.WEIGHT_DECAY = 0.00004
# 训练开始epoch数，默认为1
cfg.SOLVER.BEGIN_EPOCH = 1
# 训练epoch数，正整数
cfg.SOLVER.NUM_EPOCHS = 30
# loss的选择，支持softmax_loss, bce_loss, dice_loss
cfg.SOLVER.LOSS = ["softmax_loss"]
# loss的权重，用于多loss组合加权使用，仅对SOLVER.LOSS内包含的loss生效
cfg.SOLVER.LOSS_WEIGHT.SOFTMAX_LOSS = 1
cfg.SOLVER.LOSS_WEIGHT.DICE_LOSS = 1
cfg.SOLVER.LOSS_WEIGHT.BCE_LOSS = 1
cfg.SOLVER.LOSS_WEIGHT.LOVASZ_HINGE_LOSS = 1
cfg.SOLVER.LOSS_WEIGHT.LOVASZ_SOFTMAX_LOSS = 1
# 是否开启warmup学习策略
cfg.SOLVER.LR_WARMUP = False
# warmup的迭代次数
cfg.SOLVER.LR_WARMUP_STEPS = 2000
# cross entropy weight, 默认为None，如果设置为'dynamic'，会根据每个batch中各个类别的数目，
# 动态调整类别权重。
# 也可以设置一个静态权重(list的方式)，比如有3类，每个类别权重可以设置为[0.1, 2.0, 0.9]
cfg.SOLVER.CROSS_ENTROPY_WEIGHT = None
########################## 测试配置 ###########################################
# 测试模型路径
cfg.TEST.TEST_MODEL = ''

########################## 模型通用配置 #######################################
# 模型名称, 已支持deeplabv3p, unet, icnet，pspnet，hrnet
cfg.MODEL.MODEL_NAME = ''
# BatchNorm类型: bn、gn(group_norm)
cfg.MODEL.DEFAULT_NORM_TYPE = 'bn'
# 多路损失加权值
cfg.MODEL.MULTI_LOSS_WEIGHT = [1.0]
# DEFAULT_NORM_TYPE为gn时group数
cfg.MODEL.DEFAULT_GROUP_NUMBER = 32
# 极小值, 防止分母除0溢出，一般无需改动
cfg.MODEL.DEFAULT_EPSILON = 1e-5
# BatchNorm动量, 一般无需改动
cfg.MODEL.BN_MOMENTUM = 0.99
# 是否使用FP16训练
cfg.MODEL.FP16 = False
# 混合精度训练需对LOSS进行scale, 默认为动态scale，静态scale可以设置为512.0
cfg.MODEL.SCALE_LOSS = "DYNAMIC"

########################## DeepLab模型配置 ####################################
# DeepLab backbone 配置, 可选项xception_65, xception_41, xception_71, mobilenetv2, resnet50_vd, resnet101_vd
cfg.MODEL.DEEPLAB.BACKBONE = "xception_65"
# DeepLab output stride
cfg.MODEL.DEEPLAB.OUTPUT_STRIDE = 16
# MobileNet v2/v3 backbone scale 设置
cfg.MODEL.DEEPLAB.DEPTH_MULTIPLIER = 1.0
# DeepLab Encoder 设置
cfg.MODEL.DEEPLAB.ENCODER_WITH_ASPP = True
cfg.MODEL.DEEPLAB.ENCODER.POOLING_STRIDE = [1, 1]
cfg.MODEL.DEEPLAB.ENCODER.POOLING_CROP_SIZE = None
cfg.MODEL.DEEPLAB.ENCODER.ASPP_WITH_SE = False
cfg.MODEL.DEEPLAB.ENCODER.SE_USE_QSIGMOID = False
cfg.MODEL.DEEPLAB.ENCODER.ASPP_CONVS_FILTERS = 256
cfg.MODEL.DEEPLAB.ENCODER.ASPP_WITH_CONCAT_PROJECTION = True
cfg.MODEL.DEEPLAB.ENCODER.ADD_IMAGE_LEVEL_FEATURE = True
cfg.MODEL.DEEPLAB.ENCODER.ASPP_RATIOS = None
# DeepLab Decoder 设置
cfg.MODEL.DEEPLAB.ENABLE_DECODER = True
cfg.MODEL.DEEPLAB.DECODER.USE_SUM_MERGE = False
cfg.MODEL.DEEPLAB.DECODER.CONV_FILTERS = 256
cfg.MODEL.DEEPLAB.DECODER.OUTPUT_IS_LOGITS = False
cfg.MODEL.DEEPLAB.DECODER.ACT = True
# ASPP是否使用可分离卷积
cfg.MODEL.DEEPLAB.ASPP_WITH_SEP_CONV = True
# 解码器是否使用可分离卷积
cfg.MODEL.DEEPLAB.DECODER_USE_SEP_CONV = True
# Backbone分阶段学习率
cfg.MODEL.DEEPLAB.BACKBONE_LR_MULT_LIST = None
cfg.MODEL.DEEPLAB.BIAS = False
cfg.MODEL.DEEPLAB.ALIGN_CORNERS = True
cfg.MODEL.DEEPLAB.BENCHMARK = False

########################## UNET模型配置 #######################################
# 上采样方式, 默认为双线性插值
cfg.MODEL.UNET.UPSAMPLE_MODE = 'bilinear'

########################## ICNET模型配置 ######################################
# RESNET backbone scale 设置
cfg.MODEL.ICNET.DEPTH_MULTIPLIER = 0.5
# RESNET 层数 设置
cfg.MODEL.ICNET.LAYERS = 50

########################## PSPNET模型配置 ######################################
# RESNET backbone scale 设置
cfg.MODEL.PSPNET.DEPTH_MULTIPLIER = 1
# RESNET backbone 层数 设置
cfg.MODEL.PSPNET.LAYERS = 50

########################## HRNET模型配置 ######################################
# HRNET STAGE2 设置
cfg.MODEL.HRNET.STAGE2.NUM_MODULES = 1
cfg.MODEL.HRNET.STAGE2.NUM_CHANNELS = [40, 80]
# HRNET STAGE3 设置
cfg.MODEL.HRNET.STAGE3.NUM_MODULES = 4
cfg.MODEL.HRNET.STAGE3.NUM_CHANNELS = [40, 80, 160]
# HRNET STAGE4 设置
cfg.MODEL.HRNET.STAGE4.NUM_MODULES = 3
cfg.MODEL.HRNET.STAGE4.NUM_CHANNELS = [40, 80, 160, 320]
# FCN Head的卷积是否用bias
cfg.MODEL.HRNET.BIAS = False
cfg.MODEL.HRNET.ALIGN_CORNERS = True
########################## OCNET模型配置 ######################################

cfg.MODEL.OCR.OCR_MID_CHANNELS = 512
cfg.MODEL.OCR.OCR_KEY_CHANNELS = 256
########################## 预测部署模型配置 ###################################
# 预测保存的模型名称
cfg.FREEZE.MODEL_FILENAME = '__model__'
# 预测保存的参数名称
cfg.FREEZE.PARAMS_FILENAME = '__params__'
# 预测模型参数保存的路径
cfg.FREEZE.SAVE_DIR = 'freeze_model'

########################## paddle-slim ######################################
cfg.SLIM.KNOWLEDGE_DISTILL_IS_TEACHER = False
cfg.SLIM.KNOWLEDGE_DISTILL = False
cfg.SLIM.KNOWLEDGE_DISTILL_TEACHER_MODEL_DIR = ""

cfg.SLIM.NAS_PORT = 23333
cfg.SLIM.NAS_ADDRESS = ""
cfg.SLIM.NAS_SEARCH_STEPS = 100
cfg.SLIM.NAS_START_EVAL_EPOCH = 0
cfg.SLIM.NAS_IS_SERVER = True
cfg.SLIM.NAS_SPACE_NAME = ""

cfg.SLIM.PRUNE_PARAMS = ''
cfg.SLIM.PRUNE_RATIOS = []
cfg.SLIM.PREPROCESS = False

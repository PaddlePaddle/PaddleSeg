# 脚本使用和配置说明

PaddleSeg提供了 **训练**/**评估**/**可视化**/**模型导出** 等4个功能的使用脚本。所有脚本都支持通过不同的Flags来开启特定功能，也支持通过Options来修改默认的训练配置。它们的使用方式非常接近，如下：

```shell
# 训练
python pdseg/train.py ${FLAGS} ${OPTIONS}
# 评估
python pdseg/eval.py ${FLAGS} ${OPTIONS}
# 可视化
python pdseg/vis.py ${FLAGS} ${OPTIONS}
# 模型导出
python pdseg/export_model.py ${FLAGS} ${OPTIONS}
```

**Note:** FLAGS必须位于OPTIONS之前，否会将会遇到报错，例如如下的例子:

```shell
# FLAGS "--cfg configs/unet_optic.yaml" 必须在 OPTIONS "BATCH_SIZE 1" 之前
python pdseg/train.py BATCH_SIZE 1 --cfg configs/unet_optic.yaml
```

## 命令行FLAGS

|FLAG|用途|支持脚本|默认值|备注|
|-|-|-|-|-|
|--cfg|配置文件路径|ALL|None||
|--use_gpu|是否使用GPU进行训练|train/eval/vis|False||
|--use_mpio|是否使用多进程进行IO处理|train/eval|False|打开该开关会占用一定量的CPU内存，但是可以提高训练速度。</br> **NOTE：** windows平台下不支持该功能, 建议使用自定义数据初次训练时不打开，打开会导致数据读取异常不可见。 |
|--use_vdl|是否使用VisualDL记录训练数据|train|False||
|--log_steps|训练日志的打印周期（单位为step）|train|10||
|--debug|是否打印debug信息|train|False|IOU等指标涉及到混淆矩阵的计算，会降低训练速度|
|--vdl_log_dir &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|VisualDL的日志路径|train|None||
|--do_eval|是否在保存模型时进行效果评估   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|train|False||
|--vis_dir|保存可视化图片的路径|vis|"visual"||

## OPTIONS

PaddleSeg提供了统一的配置用于 训练/评估/可视化/导出模型。一共存在三套配置方案:
* 命令行窗口传递的参数。
* configs目录下的yaml文件。
* 默认参数，位于pdseg/utils/config.py。

三者的优先级顺序为 命令行窗口 > yaml > 默认配置。

配置包含以下Group：

|OPTIONS|用途|支持脚本|
|-|-|-|
|[BASIC](./configs/basic_group.md)|通用配置|ALL|
|[DATASET](./configs/dataset_group.md)|数据集相关|train/eval/vis|
|[MODEL](./configs/model_group.md)|模型相关|ALL|
|[TRAIN](./configs/train_group.md)|训练相关|train|
|[SOLVER](./configs/solver_group.md)|训练优化相关|train|
|[TEST](./configs/test_group.md)|测试模型相关|eval/vis/export_model|
|[AUG](./data_aug.md)|数据增强|ALL|
[FREEZE](./configs/freeze_group.md)|模型导出相关|export_model|
|[DATALOADER](./configs/dataloader_group.md)|数据加载相关|ALL|

在进行自定义的分割任务之前，您需要准备一份yaml文件，建议参照[configs目录下的示例yaml](../configs)进行修改。

以下是PaddleSeg的默认配置，供查询使用。

```yaml
########################## 基本配置 ###########################################
# 批处理大小
BATCH_SIZE: 1
# 验证时图像裁剪尺寸（宽，高）
EVAL_CROP_SIZE: tuple()
# 训练时图像裁剪尺寸（宽，高）
TRAIN_CROP_SIZE: tuple()

########################## 数据集配置 #########################################
DATASET:
    # 数据主目录目录
    DATA_DIR: './dataset/cityscapes/'
    # 训练集列表
    TRAIN_FILE_LIST: './dataset/cityscapes/train.list'
    # 验证集列表
    VAL_FILE_LIST: './dataset/cityscapes/val.list'
    # 测试数据列表
    TEST_FILE_LIST: './dataset/cityscapes/test.list'
    # VisualDL 可视化的数据集
    VIS_FILE_LIST: None
    # 类别数(需包括背景类)
    NUM_CLASSES: 19
    # 输入图像类型, 支持三通道'rgb',四通道'rgba',单通道灰度图'gray'
    IMAGE_TYPE: 'rgb'
    # 输入图片的通道数
    DATA_DIM: 3
    # 数据列表分割符, 默认为空格
    SEPARATOR: ' '
    # 忽略的像素标签值, 默认为255，一般无需改动
    IGNORE_INDEX: 255

########################## 模型通用配置 #######################################
MODEL:
    # 模型名称, 已支持deeplabv3p, unet, icnet，pspnet，hrnet
    MODEL_NAME: ''
    # BatchNorm类型: bn、gn(group_norm)
    DEFAULT_NORM_TYPE: 'bn'
    # 多路损失加权值
    MULTI_LOSS_WEIGHT: [1.0]
    # DEFAULT_NORM_TYPE为gn时group数
    DEFAULT_GROUP_NUMBER: 32
    # 极小值, 防止分母除0溢出，一般无需改动
    DEFAULT_EPSILON: 1e-5
    # BatchNorm动量, 一般无需改动
    BN_MOMENTUM: 0.99
    # 是否使用FP16训练
    FP16: False

    ########################## DeepLab模型配置 ####################################
    DEEPLAB:
        # DeepLab backbone 配置, 可选项xception_65, mobilenetv2
        BACKBONE: "xception_65"
        # DeepLab output stride
        OUTPUT_STRIDE: 16
        # MobileNet v2 backbone scale 设置
        DEPTH_MULTIPLIER: 1.0
        # MobileNet v2 backbone scale 设置
        ENCODER_WITH_ASPP: True
        # MobileNet v2 backbone scale 设置
        ENABLE_DECODER: True
        # ASPP是否使用可分离卷积
        ASPP_WITH_SEP_CONV: True
        # 解码器是否使用可分离卷积
        DECODER_USE_SEP_CONV: True

    ########################## UNET模型配置 #######################################
    UNET:
        # 上采样方式, 默认为双线性插值
        UPSAMPLE_MODE: 'bilinear'

    ########################## ICNET模型配置 ######################################
    ICNET:
        # RESNET backbone scale 设置
        DEPTH_MULTIPLIER: 0.5
        # RESNET 层数 设置
        LAYERS: 50

    ########################## PSPNET模型配置 ######################################
    PSPNET:
        # RESNET backbone scale 设置
        DEPTH_MULTIPLIER: 1
        # RESNET backbone 层数 设置
        LAYERS: 50

    ########################## HRNET模型配置 ######################################
    HRNET:
        # HRNET STAGE2 设置
        STAGE2:
            NUM_MODULES: 1
            NUM_CHANNELS: [40, 80]
        # HRNET STAGE3 设置
        STAGE3:
            NUM_MODULES: 4
            NUM_CHANNELS: [40, 80, 160]
        # HRNET STAGE4 设置
        STAGE4:
            NUM_MODULES: 3
            NUM_CHANNELS: [40, 80, 160, 320]

########################### 训练配置 ##########################################
TRAIN:
    # 模型保存路径
    MODEL_SAVE_DIR: ''
    # 预训练模型路径
    PRETRAINED_MODEL_DIR: ''
    # 是否resume，继续训练
    RESUME_MODEL_DIR: ''
    # 是否使用多卡间同步BatchNorm均值和方差
    SYNC_BATCH_NORM: False
    # 模型参数保存的epoch间隔数，可用来继续训练中断的模型
    SNAPSHOT_EPOCH: 10

########################### 模型优化相关配置 ##################################
SOLVER:
    # 初始学习率
    LR: 0.1
    # 学习率下降方法, 支持poly piecewise cosine 三种
    LR_POLICY: "poly"
    # 优化算法, 支持SGD和Adam两种算法
    OPTIMIZER: "sgd"
    # 动量参数
    MOMENTUM: 0.9
    # 二阶矩估计的指数衰减率
    MOMENTUM2: 0.999
    # 学习率Poly下降指数
    POWER: 0.9
    # step下降指数
    GAMMA: 0.1
    # step下降间隔
    DECAY_EPOCH: [10, 20]
    # 学习率权重衰减，0-1
    WEIGHT_DECAY: 0.00004
    # 训练开始epoch数，默认为1
    BEGIN_EPOCH: 1
    # 训练epoch数，正整数
    NUM_EPOCHS: 30
    # loss的选择，支持softmax_loss, bce_loss, dice_loss
    LOSS: ["softmax_loss"]
    # 是否开启warmup学习策略
    LR_WARMUP: False
    # warmup的迭代次数
    LR_WARMUP_STEPS: 2000

########################## 测试配置 ###########################################
TEST:
    # 测试模型路径
    TEST_MODEL: ''

########################### 数据增强配置 ######################################
AUG:
    # 图像resize的方式有三种：
    # unpadding（固定尺寸），stepscaling（按比例resize），rangescaling（长边对齐）
    AUG_METHOD: 'unpadding'

    # 图像resize的固定尺寸（宽，高），非负
    FIX_RESIZE_SIZE: (500, 500)

    # 图像resize方式为stepscaling，resize最小尺度，非负
    MIN_SCALE_FACTOR: 0.5
    # 图像resize方式为stepscaling，resize最大尺度，不小于MIN_SCALE_FACTOR
    MAX_SCALE_FACTOR: 2.0
    # 图像resize方式为stepscaling，resize尺度范围间隔，非负
    SCALE_STEP_SIZE: 0.25

    # 图像resize方式为rangescaling，训练时长边resize的范围最小值，非负
    MIN_RESIZE_VALUE: 400
    # 图像resize方式为rangescaling，训练时长边resize的范围最大值，
    # 不小于MIN_RESIZE_VALUE
    MAX_RESIZE_VALUE: 600
    # 图像resize方式为rangescaling, 测试验证可视化模式下长边resize的长度，
    # 在MIN_RESIZE_VALUE到MAX_RESIZE_VALUE范围内
    INF_RESIZE_VALUE: 500

    # 图像镜像左右翻转
    MIRROR: True
    # 图像上下翻转开关，True/False
    FLIP: False
    # 图像启动上下翻转的概率，0-1
    FLIP_RATIO: 0.5

    RICH_CROP:
        # RichCrop数据增广开关，用于提升模型鲁棒性
        ENABLE: False
        # 图像旋转最大角度，0-90
        MAX_ROTATION: 15
        # 裁取图像与原始图像面积比，0-1
        MIN_AREA_RATIO: 0.5
        # 裁取图像宽高比范围，非负
        ASPECT_RATIO: 0.33
        # 亮度调节范围，0-1
        BRIGHTNESS_JITTER_RATIO: 0.5
        # 饱和度调节范围，0-1
        SATURATION_JITTER_RATIO: 0.5
        # 对比度调节范围，0-1
        CONTRAST_JITTER_RATIO: 0.5
        # 图像模糊开关，True/False
        BLUR: False
        # 图像启动模糊百分比，0-1
        BLUR_RATIO: 0.1

########################## 预测部署模型配置 ###################################
FREEZE:
    # 预测保存的模型名称
    MODEL_FILENAME: '__model__'
    # 预测保存的参数名称
    PARAMS_FILENAME: '__params__'
    # 预测模型参数保存的路径
    SAVE_DIR: 'freeze_model'

########################## 数据载入配置 #######################################
DATALOADER:
    # 数据载入时的并发数, 建议值8
    NUM_WORKERS: 8
    # 数据载入时缓存队列大小, 建议值256
    BUF_SIZE: 256
```

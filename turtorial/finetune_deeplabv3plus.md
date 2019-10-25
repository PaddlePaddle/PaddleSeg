# DeepLabv3+模型训练教程

* 本教程旨在介绍如何通过使用PaddleSeg提供的 ***`DeeplabV3+/Xception65/BatchNorm`*** 预训练模型在自定义数据集上进行训练。除了该配置之外，DeeplabV3+还支持以下不同[模型组合](#模型组合)的预训练模型，如果需要使用对应模型作为预训练模型，将下述内容中的Xception Backbone中的内容进行替换即可

* 在阅读本教程前，请确保您已经了解过PaddleSeg的[快速入门](../README.md#快速入门)和[基础功能](../README.md#基础功能)等章节，以便对PaddleSeg有一定的了解

* 本教程的所有命令都基于PaddleSeg主目录进行执行

## 一. 准备待训练数据

我们提前准备好了一份数据集，通过以下代码进行下载

```shell
python dataset/download_pet.py
```

## 二. 下载预训练模型

关于PaddleSeg支持的所有预训练模型的列表，我们可以从[模型组合](#模型组合)中查看我们所需模型的名字和配置

接着下载对应的预训练模型

```shell
python pretrained_model/download_model.py deeplabv3p_xception65_bn_coco
```

## 三. 准备配置

接着我们需要确定相关配置，从本教程的角度，配置分为三部分：

* 数据集
  * 训练集主目录
  * 训练集文件列表
  * 测试集文件列表
  * 评估集文件列表
* 预训练模型
  * 预训练模型名称
  * 预训练模型的backbone网络
  * 预训练模型的Normalization类型
  * 预训练模型路径
* 其他
  * 学习率
  * Batch大小
  * ...

在三者中，预训练模型的配置尤为重要，如果模型或者BACKBONE配置错误，会导致预训练的参数没有加载，进而影响收敛速度。预训练模型相关的配置如第二步所展示。

数据集的配置和数据路径有关，在本教程中，数据存放在`dataset/mini_pet`中

其他配置则根据数据集和机器环境的情况进行调节，最终我们保存一个如下内容的yaml配置文件，存放路径为**configs/deeplabv3p_xception65_pet.yaml**

```yaml
# 数据集配置
DATASET:
    DATA_DIR: "./dataset/mini_pet/"
    NUM_CLASSES: 3
    TEST_FILE_LIST: "./dataset/mini_pet/file_list/test_list.txt"
    TRAIN_FILE_LIST: "./dataset/mini_pet/file_list/train_list.txt"
    VAL_FILE_LIST: "./dataset/mini_pet/file_list/val_list.txt"
    VIS_FILE_LIST: "./dataset/mini_pet/file_list/test_list.txt"

# 预训练模型配置
MODEL:
    MODEL_NAME: "deeplabv3p"
    DEFAULT_NORM_TYPE: "bn"
    DEEPLAB:
        BACKBONE: "xception_65"

# 其他配置
TRAIN_CROP_SIZE: (512, 512)
EVAL_CROP_SIZE: (512, 512)
AUG:
    AUG_METHOD: "unpadding"
    FIX_RESIZE_SIZE: (512, 512)
BATCH_SIZE: 4
TRAIN:
    PRETRAINED_MODEL_DIR: "./pretrained_model/deeplabv3p_xception65_bn_coco/"
    MODEL_SAVE_DIR: "./saved_model/deeplabv3p_xception65_bn_pet/"
    SNAPSHOT_EPOCH: 10
TEST:
    TEST_MODEL: "./saved_model/deeplabv3p_xception65_bn_pet/final"
SOLVER:
    NUM_EPOCHS: 100
    LR: 0.005
    LR_POLICY: "poly"
    OPTIMIZER: "sgd"
```

## 四. 配置/数据校验

在开始训练和评估之前，我们还需要对配置和数据进行一次校验，确保数据和配置是正确的。使用下述命令启动校验流程

```shell
python pdseg/check.py --cfg ./configs/deeplabv3p_xception65_pet.yaml
```


## 五. 开始训练

校验通过后，使用下述命令启动训练

```shell
python pdseg/train.py --use_gpu --cfg ./configs/deeplabv3p_xception65_pet.yaml
```

## 六. 进行评估

模型训练完成，使用下述命令启动评估

```shell
python pdseg/eval.py --use_gpu --cfg ./configs/deeplabv3p_xception65_pet.yaml
```

## 模型组合

|预训练模型名称|BackBone|Norm Type|数据集|配置|
|-|-|-|-|-|
|mobilenetv2-2-0_bn_imagenet|-|bn|ImageNet|MODEL.MODEL_NAME: deeplabv3p <br> MODEL.DEEPLAB.BACKBONE: mobilenet <br> MODEL.DEEPLAB.DEPTH_MULTIPLIER: 2.0 <br> MODEL.DEFAULT_NORM_TYPE: bn|
|mobilenetv2-1-5_bn_imagenet|-|bn|ImageNet|MODEL.MODEL_NAME: deeplabv3p <br> MODEL.DEEPLAB.BACKBONE: mobilenet <br> MODEL.DEEPLAB.DEPTH_MULTIPLIER: 1.5 <br> MODEL.DEFAULT_NORM_TYPE: bn|
|mobilenetv2-1-0_bn_imagenet|-|bn|ImageNet|MODEL.MODEL_NAME: deeplabv3p <br> MODEL.DEEPLAB.BACKBONE: mobilenet <br> MODEL.DEEPLAB.DEPTH_MULTIPLIER: 1.0 <br> MODEL.DEFAULT_NORM_TYPE: bn|
|mobilenetv2-0-5_bn_imagenet|-|bn|ImageNet|MODEL.MODEL_NAME: deeplabv3p <br> MODEL.DEEPLAB.BACKBONE: mobilenet <br> MODEL.DEEPLAB.DEPTH_MULTIPLIER: 0.5 <br> MODEL.DEFAULT_NORM_TYPE: bn|
|mobilenetv2-0-25_bn_imagenet|-|bn|ImageNet|MODEL.MODEL_NAME: deeplabv3p <br> MODEL.DEEPLAB.BACKBONE: mobilenet <br> MODEL.DEEPLAB.DEPTH_MULTIPLIER: 0.25 <br> MODEL.DEFAULT_NORM_TYPE: bn|
|xception41_imagenet|-|bn|ImageNet|MODEL.MODEL_NAME: deeplabv3p <br> MODEL.DEEPLAB.BACKBONE: xception_41 <br> MODEL.DEFAULT_NORM_TYPE: bn|
|xception65_imagenet|-|bn|ImageNet|MODEL.MODEL_NAME: deeplabv3p <br> MODEL.DEEPLAB.BACKBONE: xception_65 <br> MODEL.DEFAULT_NORM_TYPE: bn|
|deeplabv3p_mobilenetv2-1-0_bn_coco|MobileNet V2|bn|COCO|MODEL.MODEL_NAME: deeplabv3p <br> MODEL.DEEPLAB.BACKBONE: mobilenet <br> MODEL.DEEPLAB.DEPTH_MULTIPLIER: 1.0 <br> MODEL.DEEPLAB.ENCODER_WITH_ASPP: False <br> MODEL.DEEPLAB.ENABLE_DECODER: False <br> MODEL.DEFAULT_NORM_TYPE: bn|
|**deeplabv3p_xception65_bn_coco**|Xception|bn|COCO|MODEL.MODEL_NAME: deeplabv3p <br> MODEL.DEEPLAB.BACKBONE: xception_65 <br> MODEL.DEFAULT_NORM_TYPE: bn |
|deeplabv3p_mobilenetv2-1-0_bn_cityscapes|MobileNet V2|bn|Cityscapes|MODEL.MODEL_NAME: deeplabv3p <br> MODEL.DEEPLAB.BACKBONE: mobilenet <br> MODEL.DEEPLAB.DEPTH_MULTIPLIER: 1.0 <br> MODEL.DEEPLAB.ENCODER_WITH_ASPP: False <br> MODEL.DEEPLAB.ENABLE_DECODER: False <br> MODEL.DEFAULT_NORM_TYPE: bn|
|deeplabv3p_xception65_gn_cityscapes|Xception|gn|Cityscapes|MODEL.MODEL_NAME: deeplabv3p <br>  MODEL.DEEPLAB.BACKBONE: xception_65 <br> MODEL.DEFAULT_NORM_TYPE: gn|
|deeplabv3p_xception65_bn_cityscapes|Xception|bn|Cityscapes|MODEL.MODEL_NAME: deeplabv3p <br> MODEL.DEEPLAB.BACKBONE: xception_65 <br> MODEL.DEFAULT_NORM_TYPE: bn|

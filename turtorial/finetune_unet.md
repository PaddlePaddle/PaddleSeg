# U-Net模型使用教程

本教程旨在介绍如何通过使用PaddleSeg提供的 ***`U-Net`*** 预训练模型在自定义数据集上进行训练、评估和可视化。

* 在阅读本教程前，请确保您已经了解过PaddleSeg的[快速入门](../README.md#快速入门)和[基础功能](../README.md#基础功能)等章节，以便对PaddleSeg有一定的了解。

* 本教程的所有命令都基于PaddleSeg主目录进行执行。

## 一. 准备待训练数据

![](./imgs/optic.png)

我们提前准备好了一份眼底医疗分割数据集，包含267张训练图片、76张验证图片、38张测试图片。通过以下命令进行下载：

```shell
python dataset/download_optic.py
```

## 二. 下载预训练模型

接着下载对应的预训练模型

```shell
python pretrained_model/download_model.py unet_bn_coco
```

关于已有的U-Net预训练模型的列表，请参见[模型组合](#模型组合)。如果需要使用其他预训练模型，下载该模型并将配置中的BACKBONE、NORM_TYPE等进行替换即可。

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

数据集的配置和数据路径有关，在本教程中，数据存放在`dataset/optic_disc_seg`中

其他配置则根据数据集和机器环境的情况进行调节，最终我们保存一个如下内容的yaml配置文件，存放路径为**configs/unet_optic.yaml**

```yaml
# 数据集配置
DATASET:
    DATA_DIR: "./dataset/optic_disc_seg/"
    NUM_CLASSES: 2
    TEST_FILE_LIST: "./dataset/optic_disc_seg/test_list.txt"
    TRAIN_FILE_LIST: "./dataset/optic_disc_seg/train_list.txt"
    VAL_FILE_LIST: "./dataset/optic_disc_seg/val_list.txt"
    VIS_FILE_LIST: "./dataset/optic_disc_seg/test_list.txt"

# 预训练模型配置
MODEL:
    MODEL_NAME: "unet"
    DEFAULT_NORM_TYPE: "bn"

# 其他配置
TRAIN_CROP_SIZE: (512, 512)
EVAL_CROP_SIZE: (512, 512)
AUG:
    AUG_METHOD: "unpadding"
    FIX_RESIZE_SIZE: (512, 512)
BATCH_SIZE: 4
TRAIN:
    PRETRAINED_MODEL_DIR: "./pretrained_model/unet_bn_coco/"
    MODEL_SAVE_DIR: "./saved_model/unet_optic/"
    SNAPSHOT_EPOCH: 5
TEST:
    TEST_MODEL: "./saved_model/unet_optic/final"
SOLVER:
    NUM_EPOCHS: 10
    LR: 0.001
    LR_POLICY: "poly"
    OPTIMIZER: "adam"
```

## 四. 配置/数据校验

在开始训练和评估之前，我们还需要对配置和数据进行一次校验，确保数据和配置是正确的。使用下述命令启动校验流程

```shell
python pdseg/check.py --cfg ./configs/unet_optic.yaml
```


## 五. 开始训练

校验通过后，使用下述命令启动训练

```shell
# 指定GPU卡号（以0号卡为例）
export CUDA_VISIBLE_DEVICES=0
# 训练
python pdseg/train.py --use_gpu --cfg ./configs/unet_optic.yaml
```

## 六. 进行评估

模型训练完成，使用下述命令启动评估

```shell
python pdseg/eval.py --use_gpu --cfg ./configs/unet_optic.yaml
```

## 七. 进行可视化
使用下述命令启动预测和可视化

```shell
python pdseg/vis.py --use_gpu --cfg ./configs/unet_optic.yaml
```

预测结果将保存在visual目录下，以下展示其中1张图片的预测效果：

![](imgs/optic_unet.png)

## 在线体验

PaddleSeg在AI Studio平台上提供了在线体验的U-Net分割教程，欢迎[点击体验](https://aistudio.baidu.com/aistudio/projectDetail/102889)。

## 模型组合

|预训练模型名称|Backbone|数据集|配置|
|-|-|-|-|
|unet_bn_coco|VGG16|COCO|MODEL.MODEL_NAME: unet <br> MODEL.DEFAULT_NORM_TYPE: bn|

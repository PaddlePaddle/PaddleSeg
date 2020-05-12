# PaddleSeg 图像分割库

[![Build Status](https://travis-ci.org/PaddlePaddle/PaddleSeg.svg?branch=master)](https://travis-ci.org/PaddlePaddle/PaddleSeg)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleSeg.svg)](https://github.com/PaddlePaddle/PaddleSeg/releases)

## 简介

PaddleSeg是基于[PaddlePaddle](https://www.paddlepaddle.org.cn)开发的端到端图像分割开发套件，覆盖了DeepLabv3+, U-Net, ICNet, PSPNet, HRNet, Fast-SCNN等主流分割网络。通过模块化的设计，以配置化方式驱动模型组合，帮助用户更便捷地完成从训练到部署的全流程图像分割应用。

</br>

- [特点](#特点) 
- [安装](#安装)
- [使用教程](#使用教程)
  - [快速入门](#快速入门)
  - [基础功能](#基础功能)
  - [预测部署](#预测部署)
  - [高级功能](#高级功能)
- [在线体验](#在线体验)
- [FAQ](#FAQ)
- [交流与反馈](#交流与反馈)
- [更新日志](#更新日志)
- [贡献代码](#贡献代码)

</br>

## 特点

- **丰富的数据增强**

基于百度视觉技术部的实际业务经验，内置10+种数据增强策略，可结合实际业务场景进行定制组合，提升模型泛化能力和鲁棒性。

- **模块化设计**

支持U-Net, DeepLabv3+, ICNet, PSPNet, HRNet, Fast-SCNN六种主流分割网络，结合预训练模型和可调节的骨干网络，满足不同性能和精度的要求；选择不同的损失函数如Dice Loss, BCE Loss等方式可以强化小目标和不均衡样本场景下的分割精度。

- **高性能**

PaddleSeg支持多进程I/O、多卡并行、跨卡Batch Norm同步等训练加速策略，结合飞桨核心框架的显存优化功能，可大幅度减少分割模型的显存开销，让开发者更低成本、更高效地完成图像分割训练。

- **工业级部署**

全面提供**服务端**和**移动端**的工业级部署能力，依托飞桨高性能推理引擎和高性能图像处理实现，开发者可以轻松完成高性能的分割模型部署和集成。通过[Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite)，可以在移动设备或者嵌入式设备上完成轻量级、高性能的人像分割模型部署。

- **产业实践案例**

PaddleSeg提供丰富地产业实践案例，如[人像分割](./contrib/HumanSeg)、[工业表计检测](./contrib/MechanicalIndustryMeter)、[遥感分割](./contrib/RemoteSensing)、[人体解析]c(ontrib/ACE2P)，[工业质检](https://aistudio.baidu.com/aistudio/projectdetail/184392)等产业实践案例，助力开发者更便捷地落地图像分割技术。

## 安装

### 1. 安装PaddlePaddle

版本要求
* PaddlePaddle >= 1.7.0
* Python >= 3.5+

由于图像分割模型计算开销大，推荐在GPU版本的PaddlePaddle下使用PaddleSeg.
```
pip install -U paddlepaddle-gpu
```
同时请保证您参考NVIDIA官网，已经正确配置和安装了显卡驱动，CUDA 9，cuDNN 7.3，NCCL2等依赖，其他更加详细的安装信息请参考：[PaddlePaddle安装说明](https://www.paddlepaddle.org.cn/install/doc/index)。

### 2. 下载PaddleSeg代码

```
git clone https://github.com/PaddlePaddle/PaddleSeg
```

### 3. 安装PaddleSeg依赖
通过以下命令安装python包依赖，请确保在该分支上至少执行过一次以下命令：
```
cd PaddleSeg
pip install -r requirements.txt
```

</br>

## 使用教程

我们提供了一系列的使用教程，来说明如何使用PaddleSeg完成语义分割模型的训练、评估、部署。

这一系列的文档被分为**快速入门**、**基础功能**、**预测部署**、**高级功能**四个部分，四个教程由浅至深地介绍PaddleSeg的设计思路和使用方法。

### 快速入门

* [PaddleSeg快速入门](./docs/usage.md)

### 基础功能

* [自定义数据的标注与准备](./docs/data_prepare.md)
* [脚本使用和配置说明](./docs/config.md)
* [数据和配置校验](./docs/check.md)
* [分割模型介绍](./docs/models.md)
* [预训练模型下载](./docs/model_zoo.md)
* [DeepLabv3+模型使用教程](./turtorial/finetune_deeplabv3plus.md)
* [U-Net模型使用教程](./turtorial/finetune_unet.md)
* [ICNet模型使用教程](./turtorial/finetune_icnet.md)
* [PSPNet模型使用教程](./turtorial/finetune_pspnet.md)
* [HRNet模型使用教程](./turtorial/finetune_hrnet.md)
* [Fast-SCNN模型使用教程](./turtorial/finetune_fast_scnn.md)

### 预测部署

* [模型导出](./docs/model_export.md)
* [Python预测](./deploy/python/)
* [C++预测](./deploy/cpp/)
* [Paddle-Lite移动端预测部署](./deploy/lite/)


### 高级功能

* [PaddleSeg的数据增强](./docs/data_aug.md)
* [如何解决二分类中类别不均衡问题](./docs/loss_select.md)
* [特色垂类模型使用](./contrib)
* [多进程训练和混合精度训练](./docs/multiple_gpus_train_and_mixed_precision_train.md)
* 使用PaddleSlim进行分割模型压缩([量化](./slim/quantization/README.md), [蒸馏](./slim/distillation/README.md), [剪枝](./slim/prune/README.md), [搜索](./slim/nas/README.md))
## 在线体验

我们在AI Studio平台上提供了在线体验的教程，欢迎体验：

|在线教程|链接|
|-|-|
|快速开始|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/100798)|
|U-Net图像分割|[点击体验](https://aistudio.baidu.com/aistudio/projectDetail/102889)|
|DeepLabv3+图像分割|[点击体验](https://aistudio.baidu.com/aistudio/projectDetail/226703)|
|工业质检（零件瑕疵检测）|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/184392)|
|人像分割|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/188833)|
|PaddleSeg特色垂类模型|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/226710)|

</br>

## FAQ

#### Q: 安装requirements.txt指定的依赖包时，部分包提示找不到？

A: 可能是pip源的问题，这种情况下建议切换为官方源，或者通过`pip install -r requirements.txt -i `指定其他源地址。

#### Q:图像分割的数据增强如何配置，Unpadding, StepScaling, RangeScaling的原理是什么？

A: 更详细数据增强文档可以参考[数据增强](./docs/data_aug.md)

#### Q: 训练时因为某些原因中断了，如何恢复训练？

A: 启动训练脚本时通过命令行覆盖TRAIN.RESUME_MODEL_DIR配置为模型checkpoint目录即可, 以下代码示例第100轮重新恢复训练：
```
python pdseg/train.py --cfg xxx.yaml TRAIN.RESUME_MODEL_DIR /PATH/TO/MODEL_CKPT/100
```

#### Q: 预测时图片过大，导致显存不足如何处理？

A: 降低Batch size，使用Group Norm策略；请注意训练过程中当`DEFAULT_NORM_TYPE`选择`bn`时，为了Batch Norm计算稳定性，batch size需要满足>=2


#### Q: 出现错误 ModuleNotFoundError: No module named 'paddle.fluid.contrib.mixed_precision'

A: 请将PaddlePaddle升级至1.5.2版本或以上。

</br>

## 交流与反馈
* 欢迎您通过[Github Issues](https://github.com/PaddlePaddle/PaddleSeg/issues)来提交问题、报告与建议
* 微信公众号：飞桨PaddlePaddle
* QQ群: 796771754

<p align="center"><img width="200" height="200"  src="https://user-images.githubusercontent.com/45189361/64117959-1969de80-cdc9-11e9-84f7-e1c2849a004c.jpeg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="200" height="200" margin="500" src="./docs/imgs/qq_group2.png"/></p>
<p align="center">  &#8194;&#8194;&#8194;微信公众号&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;官方技术交流QQ群</p>

## 更新日志
* 2020.02.25

  **`v0.4.0`**
  * 新增适用于实时场景且不需要预训练模型的分割网络Fast-SCNN，提供基于Cityscapes的[预训练模型](./docs/model_zoo.md)1个。
  * 新增LaneNet车道线检测网络，提供[预训练模型](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v0.4.0/contrib/LaneNet#%E4%B8%83-%E5%8F%AF%E8%A7%86%E5%8C%96)一个。
  * 新增基于PaddleSlim的分割库压缩策略([量化](./slim/quantization/README.md), [蒸馏](./slim/distillation/README.md), [剪枝](./slim/prune/README.md), [搜索](./slim/nas/README.md))
  
  
* 2019.12.15

  **`v0.3.0`**
  * 新增HRNet分割网络，提供基于cityscapes和ImageNet的[预训练模型](./docs/model_zoo.md)8个
  * 支持使用[伪彩色标签](./docs/data_prepare.md#%E7%81%B0%E5%BA%A6%E6%A0%87%E6%B3%A8vs%E4%BC%AA%E5%BD%A9%E8%89%B2%E6%A0%87%E6%B3%A8)进行训练/评估/预测，提升训练体验，并提供将灰度标注图转为伪彩色标注图的脚本
  * 新增[学习率warmup](./docs/configs/solver_group.md#lr_warmup)功能，支持与不同的学习率Decay策略配合使用
  * 新增图像归一化操作的GPU化实现，进一步提升预测速度。
  * 新增Python部署方案，更低成本完成工业级部署。
  * 新增Paddle-Lite移动端部署方案，支持人像分割模型的移动端部署。
  * 新增不同分割模型的预测[性能数据Benchmark](./deploy/python/docs/PaddleSeg_Infer_Benchmark.md), 便于开发者提供模型选型性能参考。

  
* 2019.11.04

  **`v0.2.0`**
  * 新增PSPNet分割网络，提供基于COCO和cityscapes数据集的[预训练模型](./docs/model_zoo.md)4个。
  * 新增Dice Loss、BCE Loss以及组合Loss配置，支持样本不均衡场景下的[模型优化](./docs/loss_select.md)。
  * 支持[FP16混合精度训练](./docs/multiple_gpus_train_and_mixed_precision_train.md)以及动态Loss Scaling，在不损耗精度的情况下，训练速度提升30%+。
  * 支持[PaddlePaddle多卡多进程训练](./docs/multiple_gpus_train_and_mixed_precision_train.md)，多卡训练时训练速度提升15%+。
  * 发布基于UNet的[工业标记表盘分割模型](./contrib#%E5%B7%A5%E4%B8%9A%E7%94%A8%E8%A1%A8%E5%88%86%E5%89%B2)。

* 2019.09.10

  **`v0.1.0`**
  * PaddleSeg分割库初始版本发布，包含DeepLabv3+, U-Net, ICNet三类分割模型, 其中DeepLabv3+支持Xception, MobileNet v2两种可调节的骨干网络。
  * CVPR19 LIP人体部件分割比赛冠军预测模型发布[ACE2P](./contrib/ACE2P)。
  * 预置基于DeepLabv3+网络的[人像分割](./contrib/HumanSeg/)和[车道线分割](./contrib/RoadLine)预测模型发布。

</br>

## 贡献代码

我们非常欢迎您为PaddleSeg贡献代码或者提供使用建议。如果您可以修复某个issue或者增加一个新功能，欢迎给我们提交pull requests.

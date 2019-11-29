# PaddleSeg 图像分割库

[![Build Status](https://travis-ci.org/PaddlePaddle/PaddleSeg.svg?branch=master)](https://travis-ci.org/PaddlePaddle/PaddleSeg)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleSeg.svg)](https://github.com/PaddlePaddle/PaddleSeg/releases)

## 简介

PaddleSeg是基于[PaddlePaddle](https://www.paddlepaddle.org.cn)开发的语义分割库，覆盖了DeepLabv3+, U-Net, ICNet三类主流的分割模型。通过统一的配置，帮助用户更便捷地完成从训练到部署的全流程图像分割应用。

PaddleSeg具备高性能、丰富的数据增强、工业级部署、全流程应用的特点:


- **丰富的数据增强**

基于百度视觉技术部的实际业务经验，内置10+种数据增强策略，可结合实际业务场景进行定制组合，提升模型泛化能力和鲁棒性。

- **模块化设计**

支持U-Net, DeepLabv3+, ICNet, PSPNet四种主流分割网络，结合预训练模型和可调节的骨干网络，满足不同性能和精度的要求；选择不同的损失函数如Dice Loss, BCE Loss等方式可以强化小目标和不均衡样本场景下的分割精度。

- **高性能**

PaddleSeg支持多进程IO、多卡并行、跨卡Batch Norm同步等训练加速策略，结合飞桨核心框架的显存优化功能，可以大幅度减少分割模型的显存开销，更快完成分割模型训练。

- **工业级部署**

基于[Paddle Serving](https://github.com/PaddlePaddle/Serving)和PaddlePaddle高性能预测引擎，结合百度开放的AI能力，轻松搭建人像分割和车道线分割服务。

</br>

## 环境依赖

* PaddlePaddle >= 1.6.1
* Python 2.7 or 3.5+

通过以下命令安装python包依赖，请确保在该分支上至少执行过一次以下命令
```shell
$ pip install -r requirements.txt
```

其他如CUDA版本、cuDNN版本等兼容信息请查看[PaddlePaddle安装](https://www.paddlepaddle.org.cn/install/doc/index)

</br>

## 使用教程

我们提供了一系列的使用教程，来说明如何使用PaddleSeg完成语义分割模型的训练、评估、部署。

这一系列的文档被分为**快速入门**、**基础功能**、**预测部署**、**高级功能**四个部分，四个教程由浅至深地介绍PaddleSeg的设计思路和使用方法。

### 快速入门

* [安装说明](./docs/installation.md)
* [训练/评估/可视化](./docs/usage.md)

### 基础功能

* [分割模型介绍](./docs/models.md)
* [预训练模型列表](./docs/model_zoo.md)
* [自定义数据的准备与标注](./docs/data_prepare.md)
* [数据和配置校验](./docs/check.md)
* [如何训练DeepLabv3+](./turtorial/finetune_deeplabv3plus.md)
* [如何训练U-Net](./turtorial/finetune_unet.md)
* [如何训练ICNet](./turtorial/finetune_icnet.md)
* [如何训练PSPNet](./turtorial/finetune_pspnet.md)
* [如何训练HRNet](./turtorial/finetune_hrnet.md)

### 预测部署

* [模型导出](./docs/model_export.md)
* [使用Python预测](./deploy/python/)
* [使用C++预测](./deploy/cpp/)
* [移动端预测部署](./deploy/lite/)


### 高级功能

* [PaddleSeg的数据增强](./docs/data_aug.md)
* [PaddleSeg的loss选择](./docs/loss_select.md)
* [特色垂类模型使用](./contrib)
* [多进程训练和混合精度训练](./docs/multiple_gpus_train_and_mixed_precision_train.md)

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

</br>

#### Q: 出现错误 ModuleNotFoundError: No module named 'paddle.fluid.contrib.mixed_precision'

A: 请将PaddlePaddle升级至1.5.2版本或以上。

## 在线体验

PaddleSeg在AI Studio平台上提供了在线体验的教程，欢迎体验：

|教程|链接|
|-|-|
|U-Net宠物分割|[点击体验](https://aistudio.baidu.com/aistudio/projectDetail/102889)|
|DeepLabv3+图像分割|[点击体验](https://aistudio.baidu.com/aistudio/projectDetail/101696)|
|PaddleSeg特色垂类模型|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/115541)|

</br>

##  交流与反馈
* 欢迎您通过[Github Issues](https://github.com/PaddlePaddle/PaddleSeg/issues)来提交问题、报告与建议
* 微信公众号：飞桨PaddlePaddle
* QQ群: 796771754

<p align="center"><img width="200" height="200"  src="https://user-images.githubusercontent.com/45189361/64117959-1969de80-cdc9-11e9-84f7-e1c2849a004c.jpeg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="200" height="200" margin="500" src="./docs/imgs/qq_group2.png"/></p>
<p align="center">  &#8194;&#8194;&#8194;微信公众号&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;官方技术交流QQ群</p>

## 更新日志

* 2019.11.04

  **`v0.2.0`**
  * 新增PSPNet分割网络，提供基于COCO和cityscapes数据集的[预训练模型](./docs/model_zoo.md)4个
  * 新增Dice Loss、BCE Loss以及组合Loss配置，支持样本不均衡场景下的[模型优化](./docs/loss_select.md)
  * 支持[FP16混合精度训练](./docs/multiple_gpus_train_and_mixed_precision_train.md)以及动态Loss Scaling，在不损耗精度的情况下，训练速度提升30%+
  * 支持[PaddlePaddle多卡多进程训练](./docs/multiple_gpus_train_and_mixed_precision_train.md)，多卡训练时训练速度提升15%+
  * 发布基于UNet的[工业标记表盘分割模型](./contrib#%E5%B7%A5%E4%B8%9A%E7%94%A8%E8%A1%A8%E5%88%86%E5%89%B2)

* 2019.09.10

  **`v0.1.0`**
  * PaddleSeg分割库初始版本发布，包含DeepLabv3+, U-Net, ICNet三类分割模型, 其中DeepLabv3+支持Xception, MobileNet v2两种可调节的骨干网络。
  * CVPR19 LIP人体部件分割比赛冠军预测模型发布[ACE2P](./contrib/ACE2P)
  * 预置基于DeepLabv3+网络的[人像分割](./contrib/HumanSeg/)和[车道线分割](./contrib/RoadLine)预测模型发布

</br>

## 如何贡献代码

我们非常欢迎您为PaddleSeg贡献代码或者提供使用建议。

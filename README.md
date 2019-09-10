# PaddleSeg 语义分割库

[![Build Status](https://travis-ci.org/PaddlePaddle/PaddleSeg.svg?branch=master)](https://travis-ci.org/PaddlePaddle/PaddleSeg)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

## 简介

PaddleSeg是基于[PaddlePaddle](https://www.paddlepaddle.org.cn)开发的语义分割库，覆盖了DeepLabv3+, U-Net, ICNet三类主流的分割模型。通过统一的配置，帮助用户更便捷地完成从训练到部署的全流程图像分割应用。

PaddleSeg具备高性能、丰富的数据增强、工业级部署、全流程应用的特点:


- **丰富的数据增强**

基于百度视觉技术部的实际业务经验，内置10+种数据增强策略，可结合实际业务场景进行定制组合，提升模型泛化能力和鲁棒性。

- **主流模型覆盖**

支持U-Net, DeepLabv3+, ICNet三类主流分割网络，结合预训练模型和可调节的骨干网络，满足不同性能和精度的要求。

- **高性能**

PaddleSeg支持多进程IO、多卡并行、跨卡Batch Norm同步等训练加速策略，结合飞桨核心框架的显存优化功能，可以大幅度减少分割模型的显存开销，更快完成分割模型训练。

- **工业级部署**

基于[Paddle Serving](https://github.com/PaddlePaddle/Serving)和PaddlePaddle高性能预测引擎，结合百度开放的AI能力，轻松搭建人像分割和车道线分割服务。

</br>

## 使用教程

我们提供了一系列的使用教程，来说明如何使用PaddleSeg完成一个语义分割模型的训练、评估、部署。

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

### 预测部署

* [模型导出](./docs/model_export.md)
* [C++预测库使用](./inference)
* [PaddleSeg Serving服务化部署](./serving)

### 高级功能

* [PaddleSeg的数据增强](./docs/data_aug.md)
* [特色垂类模型使用](./contrib)

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

## 在线体验

PaddleSeg在AI Studio平台上提供了在线体验的教程，欢迎体验：

|教程|链接|
|-|-|
|U-Net宠物分割|[点击体验](https://aistudio.baidu.com/aistudio/projectDetail/102889)|
|DeepLabv3+图像分割|[点击体验](https://aistudio.baidu.com/aistudio/projectDetail/101696)|
|PaddleSeg特色垂类模型|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/115541)|

</br>

##  交流与反馈
* 欢迎您通过Github Issues来提交问题、报告与建议
* 微信公众号：飞桨PaddlePaddle
* QQ群: 432676488 

<p align="center"><img width="200" height="200"  src="https://user-images.githubusercontent.com/45189361/64117959-1969de80-cdc9-11e9-84f7-e1c2849a004c.jpeg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="200" height="200" margin="500" src="./docs/imgs/qq_group2.png"/></p>
<p align="center">  &#8194;&#8194;&#8194;微信公众号&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;官方技术交流QQ群</p>

* 论坛: 欢迎大家在[PaddlePaddle论坛](https://ai.baidu.com/forum/topic/list/168)分享在使用PaddlePaddle中遇到的问题和经验, 营造良好的论坛氛围

## 更新日志

* 2019.09.10

  **`v0.1.0`**
  * PaddleSeg分割库初始版本发布，包含DeepLabv3+, U-Net, ICNet三类分割模型, 其中DeepLabv3+支持Xception, MobileNet两种可调节的骨干网络。
  * CVPR19 LIP人体部件分割比赛冠军预测模型发布[ACE2P](./contrib/ACE2P)
  * 预置基于DeepLabv3+网络的[人像分割](./contrib/HumanSeg/)和[车道线分割](./contrib/RoadLine)预测模型发布

</br>

## 如何贡献代码

我们非常欢迎您为PaddleSeg贡献代码或者提供使用建议。

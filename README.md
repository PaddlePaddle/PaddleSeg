# PaddleSeg 语义分割库

## 简介

PaddleSeg是基于[PaddlePaddle](https://www.paddlepaddle.org.cn)开发的语义分割库，覆盖了DeepLabv3+, U-Net, ICNet三类主流的分割模型。通过统一的配置，帮助用户更便捷地完成从训练到部署的全流程图像分割应用。
具备高性能、丰富的数据增强、工业级部署、全流程应用的特点。



- **丰富的数据增强**

  - 基于百度视觉技术部的实际业务经验，内置10+种数据增强策略，可结合实际业务场景进行定制组合，提升模型泛化能力和鲁棒性。
  
- **主流模型覆盖**

  - 支持U-Net, DeepLabv3+, ICNet三类主流分割网络，结合预训练模型和可调节的骨干网络，满足不同性能和精度的要求。

- **高性能**

  - PaddleSeg支持多进程IO、多卡并行、多卡Batch Norm, FP16混合精度等训练加速策略，通过飞桨核心框架的显存优化算法，可以大幅度节约分割模型的显存开销，更快完成分割模型训练。
  
- **工业级部署**

  - 基于[Paddle Serving](https://github.com/PaddlePaddle/Serving)和PaddlePaddle高性能预测引擎, 结合百度开放的AI能力，轻松搭建人像分割和车道线分割服务。




更多模型信息与技术细节请查看[模型介绍](./docs/models.md)和[预训练模型](./docs/mode_zoo.md)

## AI Studio教程

### 快速开始

通过 [PaddleSeg人像分割](https://aistudio.baidu.com/aistudio/projectDetail/100798) 教程可快速体验PaddleSeg人像分割模型的效果。

### 入门教程

入门教程以经典的U-Net模型为例, 结合Oxford-IIIT宠物数据集，快速熟悉PaddleSeg使用流程, 详情请点击[U-Net宠物分割](https://aistudio.baidu.com/aistudio/projectDetail/102889)。

### 高级教程

高级教程以DeepLabv3+模型为例，结合Cityscapes数据集，快速了解ASPP, Backbone网络切换，多卡Batch Norm同步等策略，详情请点击[DeepLabv3+图像分割](https://aistudio.baidu.com/aistudio/projectDetail/101696)。

### 垂类模型

更多特色垂类分割模型如LIP人体部件分割、人像分割、车道线分割模型可以参考[contrib](./contrib/README.md)

## 使用文档

* [安装说明](./docs/installation.md)
* [数据准备](./docs/data_prepare.md)
* [数据增强](./docs/data_aug.md)
* [预训练模型](./docs/model_zoo.md)
* [训练/评估/预测(可视化)](./docs/usage.md)
* [预测库集成](./inference/README.md)
* [服务端部署](./serving/README.md)
* [垂类分割模型](./contrib/README.md)


## FAQ

#### Q:图像分割的数据增强如何配置，unpadding, step scaling, range scaling的原理是什么？

A:数据增强的配置可以参考文档[数据增强](./docs/data_aug.md)

#### Q: 预测时图片过大，导致显存不足如何处理？

A: 降低Batch size，使用Group Norm策略等。

## 更新日志

### 2019.08.25

#### v0.1.0

* PaddleSeg分割库初始版本发布，包含DeepLabv3+, U-Net, ICNet三类分割模型, 其中DeepLabv3+支持Xception, MobileNet两种可调节的骨干网络。
* CVPR 19' LIP人体部件分割比赛冠军预测模型发布[ACE2P](./contrib/ACE2P)
* 预置基于DeepLabv3+网络的[人像分割](./contrib/HumanSeg/)和[车道线分割](./contrib/RoadLine)预测模型发布

## 如何贡献代码

我们非常欢迎您为PaddleSeg贡献代码或者提供使用建议。

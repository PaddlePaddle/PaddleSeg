简体中文 | [English](README.md)

# PaddleSeg

[![Build Status](https://travis-ci.org/PaddlePaddle/PaddleSeg.svg?branch=master)](https://travis-ci.org/PaddlePaddle/PaddleSeg)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleSeg.svg)](https://github.com/PaddlePaddle/PaddleSeg/releases)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)

<img src="./docs/images/seg_news_icon.png" width="50"/> *[2020-12-18] PaddleSeg发布2.0.0rc版，动态图正式成为主目录。静态图已经被移至[legacy](./legacy)子目录下。更多信息请查看详细[更新日志](./docs/release_notes_cn.md)。*

![demo](./docs/images/cityscapes.gif)

PaddleSeg是基于飞桨[PaddlePaddle](https://www.paddlepaddle.org.cn)开发的端到端图像分割开发套件，涵盖了**高精度**和**轻量级**等不同方向的大量高质量分割模型。通过模块化的设计，提供了**配置化驱动**和**API调用**等两种应用方式，帮助开发者更便捷地完成从训练到部署的全流程图像分割应用。

## 特性

* **高精度模型**：基于百度自研的[半监督标签知识蒸馏方案（SSLD）](https://paddleclas.readthedocs.io/zh_CN/latest/advanced_tutorials/distillation/distillation.html#ssld)训练得到高精度骨干网络，结合前沿的分割技术，提供了50+的高质量预训练模型，效果优于其他开源实现。

* **模块化设计**：支持15+主流 *分割网络* ，结合模块化设计的 *数据增强策略* 、*骨干网络*、*损失函数* 等不同组件，开发者可以基于实际应用场景出发，组装多样化的训练配置，满足不同性能和精度的要求。

* **高性能**：支持多进程异步I/O、多卡并行训练、评估等加速策略，结合飞桨核心框架的显存优化功能，可大幅度减少分割模型的训练开销，让开发者更低成本、更高效地完成图像分割训练。

## 模型库

|模型\骨干网络|ResNet50|ResNet101|HRNetw18|HRNetw48|
|-|-|-|-|-|
|[ANN](./configs/ann)|✔|✔|||
|[BiSeNetv2](./configs/bisenet)|-|-|-|-|
|[DANet](./configs/danet)|✔|✔|||
|[Deeplabv3](./configs/deeplabv3)|✔|✔|||
|[Deeplabv3P](./configs/deeplabv3p)|✔|✔|||
|[Fast-SCNN](./configs/fastscnn)|-|-|-|-|
|[FCN](./configs/fcn)|||✔|✔|
|[GCNet](./configs/gcnet)|✔|✔|||
|[GSCNN](./configs/gscnn)|✔|✔|||
|[HarDNet](./configs/hardnet)|-|-|-|-|
|[OCRNet](./configs/ocrnet/)|||✔|✔|
|[PSPNet](./configs/pspnet)|✔|✔|||
|[U-Net](./configs/unet)|-|-|-|-|
|[U<sup>2</sup>-Net](./configs/u2net)|-|-|-|-|
|[Att U-Net](./configs/attention_unet)|-|-|-|-|
|[U-Net++](./configs/unet_plusplus)|-|-|-|-|
|[EMANet](./configs/emanet)|✔|✔|-|-|
|[ISANet](./configs/isanet)|✔|✔|-|-|

## 数据集

- [x] Cityscapes
- [x] Pascal VOC
- [x] ADE20K
- [ ] Pascal Context
- [ ] COCO stuff

## 安装

#### 1. 安装PaddlePaddle

版本要求

* PaddlePaddle >= 2.0.0rc

* Python >= 3.6+

由于图像分割模型计算开销大，推荐在GPU版本的PaddlePaddle下使用PaddleSeg。推荐安装10.0以上的CUDA环境。安装教程请见[PaddlePaddle官网](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/install/index_cn.html)。


#### 2. 安装PaddleSeg

```shell
pip install paddleseg
```

#### 3. 下载PaddleSeg仓库
```shell
git clone https://github.com/PaddlePaddle/PaddleSeg
```

## 训练
```shell
python train.py --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml
```

## 使用教程

* [快速入门](./docs/quick_start.md)
* [API使用教程](https://aistudio.baidu.com/aistudio/projectdetail/1339458)
* [数据集准备](./docs/data_prepare.md)
* [配置项](./configs/)
* [API参考](./docs/apis)
* [添加新组件](./docs/add_new_model.md)

## 联系我们
* 如果你发现任何PaddleSeg存在的问题或者是建议, 欢迎通过[GitHub Issues](https://github.com/PaddlePaddle/PaddleSeg/issues)给我们提issues。
* 同时欢迎加入PaddleSeg技术交流群：850378321（QQ群1）或者793114768（QQ群2）。

## 代码贡献

* 非常感谢[jm12138](https://github.com/jm12138)贡献U<sup>2</sup>-Net模型。
* 非常感谢[zjhellofss](https://github.com/zjhellofss)（傅莘莘）贡献Attention U-Net模型，和Dice loss损失函数。
* 非常感谢[liuguoyu666](https://github.com/liguoyu666)贡献U-Net++模型。

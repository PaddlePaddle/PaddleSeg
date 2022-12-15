中文 | [English](doc/EN/README.md)

<div align="center">

<p align="center">
  <img src="https://user-images.githubusercontent.com/35907364/182084617-ea94f744-3a34-4193-98fe-5d6869a118fc.png" align="middle" alt="LOGO" width = "500" />
</p>

**飞桨智能标注，让标注快人一步**

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-390/) ![PyPI](https://img.shields.io/pypi/v/paddlelabel?color=blue) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Start](https://img.shields.io/github/stars/PaddleCV-SIG/PaddleLabel?color=orange)](<>) [![Fork](https://img.shields.io/github/forks/PaddleCV-SIG/PaddleLabel?color=orange)](<>) ![PyPI - Downloads](https://img.shields.io/pypi/dm/paddlelabel?color=orange) [![OS](https://img.shields.io/badge/os-linux%2C%20windows%2C%20macos-green.svg)](<>)

</div>

## 最新动态

- 【2022-11-30】 :fire: PaddleLabel 0.5 版本发布！

  - 【界面】全面升级分类、检测及分割的前端标注界面体验，显著提升标注流畅度。
  - 【分类】新增PPLCNet预训练模型，为分类功能提供预标注能力。
  - 【检测】新增PicoDet预训练模型，为检测功能提供预标注能力。
  - 【分割】(1)优化语义分割及实例分割关于实例的区分，实例分割通过'确认轮廓'来区分实例; (2)新增根据类别或根据实例选择颜色显示模式; (3)修复交互式分割localStorage超限问题。

- 【2022-08-18】 :fire: PaddleLabel 0.1 版本发布！

  - 【分类】支持单分类与多分类标注及标签的导入导出。简单灵活实现自定义数据集分类标注任务并导出供[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)进行训练。
  - 【检测】支持检测框标注及标签的导入导出。快速上手生成自己的检测数据集并应用到[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)。
  - 【分割】支持多边形、笔刷及交互式等多种标注方式，支持标注语义分割与实例分割两种场景。多种分割标注方式可灵活选择，方便将导出数据应用在[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)获取个性化定制模型。

## 简介

PaddleLabel 是基于飞桨 PaddlePaddle 各个套件的特性提供的配套标注工具。它涵盖分类、检测、分割三种常见的计算机视觉任务的标注能力，具有手动标注和交互式标注相结合的能力。用户可以使用 PaddleLabel 方便快捷的标注自定义数据集并将导出数据用于飞桨提供的其他套件的训练预测流程中。
PaddleLabel 的代码分布于三个项目中，本项目包含 PaddleLabel 的后端实现。 [PaddleLabel-Frontend](https://github.com/PaddleCV-SIG/PaddleLabel-Frontend)是基于 React 和 Ant Design 构建的 PaddleLabel 前端，[PaddleLabel-ML](https://github.com/PaddleCV-SIG/PaddleLabel-ML)是基于 PaddlePaddle 的自动和交互式机器学习辅助标注后端。

![demo720](https://user-images.githubusercontent.com/71769312/185099439-3230cf80-798d-4a81-bcae-b88bcb714daa.gif)

## 特性

- **简单** 手动标注直观易操作，上手成本极低。
- **高效** 支持交互式分割和预标注，标注精度及效率提升显著。
- **灵活** 分类支持单分类和多分类的标注，分割支持多边形、笔刷及交互式分割等多种功能，方便用户根据场景需求切换标注方式。
- **全流程** 与其他飞桨套件密切配合，方便用户完成数据标注、模型训练、模型导出等全流程操作。

## 技术交流

- 任何使用问题、产品建议、功能需求, 可以[提交 Issues](https://github.com/PaddleCV-SIG/PaddleLabel/issues/new)与开发团队交流。
- 欢迎大家扫码加入 PaddleLabel 微信群，和小伙伴们一起交流学习。

<div align="center">
<img src="https://user-images.githubusercontent.com/29757093/206070067-4e61cc56-34af-4e11-ba64-d4976e439bda.png"  width = "200" />
</div>

## 使用教程

- [安装指南](doc/CN/install.md)
- [快速开始](doc/CN/quick_start.md)

### 进行标注

- [图像分类](doc/CN/project/classification.md)
- [图像分类自动标注](doc/CN/project/classification_auto_label.md)
- [目标检测](doc/CN/project/detection.md)
- [目标检测自动标注](doc/CN/project/detection_auto_label.md)
- [语义分割](doc/CN/project/semantic_segmentation.md)
- [实例分割](doc/CN/project/instance_segmentation.md)
- [交互式分割标注](doc/CN/project/interactive_segmentation.md)

### 训练教程

- [如何用 PaddleClas 进行训练](doc/CN/training/PdLabel_PdClas.md)
- [如何用 PaddleDet 进行训练](doc/CN/training/PdLabel_PdDet.md)
- [如何使用 PaddleSeg 进行训练](doc/CN/training/PdLabel_PdSeg.md)
- [如何使用 PaddleX 进行训练](doc/CN/training/PdLabel_PdX.md)

### AI Studio 项目

- [花朵分类](https://aistudio.baidu.com/aistudio/projectdetail/4337003)
- [道路标志检测](https://aistudio.baidu.com/aistudio/projectdetail/4349280)
- [图像分割](https://aistudio.baidu.com/aistudio/projectdetail/4353528)
- [如何使用 PaddleX 进行训练](https://aistudio.baidu.com/aistudio/projectdetail/4383953)

## 社区贡献

### 贡献者

感谢下列开发者参与或协助 PaddleLabel 的开发、维护、测试等：[linhandev](https://github.com/linhandev)、[cheneyveron](https://github.com/cheneyveron)、[Youssef-Harby](https://github.com/Youssef-Harby)、[geoyee](https://github.com/geoyee)、[yzl19940819](https://github.com/yzl19940819)、[haoyuying](https://github.com/haoyuying)

### 参与开发

如果您愿意参与项目开发，请通过微信交流群联系开发团队，我们十分愿意接纳新的开发者到项目的开发和维护中。有关后端实现的详细信息，请参阅[开发者指南](doc/CN/developers_guide.md)。

<p align="right">(<a href="#top">返回顶部</a>)</p>

<!-- quote-->

## 学术引用

```
@misc{paddlelabel2022,
    title={PaddleLabel, an effective and flexible tool for data annotation},
    author={PaddlePaddle Authors},
    howpublished = {\url{https://github.com/PaddleCV-SIG/PaddleLabel}},
    year={2022}
}
```

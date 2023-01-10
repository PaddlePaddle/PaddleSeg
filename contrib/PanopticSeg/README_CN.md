[English](README_EN.md) | 简体中文

# 基于 PaddleSeg 的全景分割工具箱

## 索引

+ [简介](#简介)
+ [更新日志](#更新日志)
+ [已支持的模型](#已支持的模型)
+ [使用教程](#使用教程)
+ [社区贡献 & 技术交流](#社区贡献--技术交流)

## 简介

全景分割是一项图像解析任务，该任务结合了语义分割（为图像中每个像素赋予一个标签）和实例分割（检测并分割出图像中每一个对象实例）。本工具箱基于 [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) 打造，旨在提供全景分割模型训练、验证与部署的全流程开发解决方案。

+ **高精度**：提供高质量的前沿全景分割模型，开箱即用。
+ **高性能**：使用多进程异步I/O、多卡并行训练等加速策略，结合飞桨核心框架的显存优化功能，让开发者以更低成本、更高效地完成全景分割模型训练。
+ **全流程**：支持从模型设计到模型部署的完整工作流，助力用户完成一站式开发工作。

<p align="center">
<img src="https://user-images.githubusercontent.com/21275753/210925385-5021e2b6-2d73-4358-a9af-1e91cd9f008d.gif" height="150">
<img src="https://user-images.githubusercontent.com/21275753/210925394-57848331-0bd5-4c30-9fb0-03fc2a789936.gif" height="150">
<img src="https://user-images.githubusercontent.com/21275753/210925397-0b348fcf-b3f9-46cf-9512-b50278138658.gif" height="150">
</p>

+ *以上效果展示图基于 [Cityscapes](https://www.cityscapes-dataset.com/) 和 [MS COCO](https://cocodataset.org/#home) 数据集中的图片以及使用本工具箱训练的模型所得到的推理结果。*

## 更新日志

+ 2022.12
    - 新增 Mask2Former 与 Panoptic-DeepLab 模型。

## 已支持的模型

+ [Mask2Former](configs/mask2former/README.md)
+ [Panoptic-DeepLab](configs/panoptic_deeplab/README.md)

## 使用教程
+ [快速开始](docs/quick_start_cn.md)
+ [完整功能](docs/full_features_cn.md)
+ [开发指南](docs/dev_guide_cn.md)

## 社区贡献 & 技术交流

+ 如果大家对本工具箱有任何使用问题和功能建议，可以通过 [GitHub Issues](https://github.com/PaddlePaddle/PaddleSeg/issues) 向我们提 issue。
+ 欢迎加入PaddleSeg的微信用户群👫（扫码填写简单问卷即可入群），大家可以领取30G重磅学习大礼包🎁，也可以和值班同学、各界大佬直接进行交流。

<div align="center">
<img src="https://user-images.githubusercontent.com/48433081/174770518-e6b5319b-336f-45d9-9817-da12b1961fb1.jpg" width = "200" />  
</div>
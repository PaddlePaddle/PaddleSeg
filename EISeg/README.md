简体中文 | [English](README_EN.md)
<div align="center">

<p align="center">
  <img src="https://user-images.githubusercontent.com/35907364/179460858-7dfb19b1-cabf-4f8a-9e81-eb15b6cc7d5f.png" align="middle" alt="LOGO" width = "500" />
</p>
**飞桨高效交互式分割标注工具。**

[![Python 3.6](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/) [![PaddlePaddle 2.2](https://img.shields.io/badge/paddlepaddle-2.2-blue.svg)](https://www.python.org/downloads/release/python-360/) [![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE) [![Downloads](https://pepy.tech/badge/eiseg)](https://pepy.tech/project/eiseg)

</div>

<div align="center">
<table>
    <tr>
        <td><img src="https://user-images.githubusercontent.com/71769312/179209324-eb074e65-4a32-4568-a1d3-7680331dbf22.gif"></td>
        <td><img src="https://user-images.githubusercontent.com/71769312/179209332-e3bcb1f0-d4d9-44e1-8b2a-8d7fac8996d4.gif"></td>
        <td><img src="https://user-images.githubusercontent.com/71769312/179209312-0febfe78-810d-49b2-9169-eb15f0523af7.gif"></td>
        <td><img src="https://user-images.githubusercontent.com/71769312/179209340-d04a0cec-d9a7-4962-93f1-b4953c6c9f39.gif"></td>
    <tr>
    <tr>
        <td align="center">通用分割</td>
        <td align="center">人像分割</td>
        <td align="center">遥感建筑分割</td>
        <td align="center">医疗分割</td>
    <tr>
    <tr>
        <td><img src="https://user-images.githubusercontent.com/71769312/185751161-f23d0c1b-62c5-4cd2-903f-502037e353a8.gif"></td>
        <td><img src="https://user-images.githubusercontent.com/71769312/179209328-87174780-6c6f-4b53-b2a2-90d289ac1c8a.gif"></td>
        <td colspan="2"><img src="https://user-images.githubusercontent.com/71769312/179209342-5b75e61e-d9cf-4702-ba3e-971f47a10f5f.gif"></td>
    <tr>
    <tr>
        <td align="center">工业质检</td>
        <td align="center">通用视频分割</td>
        <td align="center" colspan="2">3D医疗分割</td>
    <tr>
</table>
</div>


## <img src="../docs/images/seg_news_icon.png" width="20"/> 最新动态
* [2022-12-16] :fire: EISeg 1.1版本发布！
  - 新增检测标注功能，可手工标注或使用预标注模型PicoDet进行标注。
  - 检测标注结果保存格式支持COCO, VOC及YOLO等多种格式。
  - 分割新增LabelMe JSON保存格式。
* [2022-09-16] :fire: EISeg使用的X光胸腔标注模型MUSCLE已经被MICCAI 2022接收，具体可参见[MUSCLE](docs/MUSCLE.md), 标注模型下载[地址](https://paddleseg.bj.bcebos.com/eiseg/0.5/static_resnet50_deeplab_chest_xray.zip).

## <img src="https://user-images.githubusercontent.com/48054808/157795569-9fc77c85-732f-4870-9be0-99a7fe2cff27.png" width="20"/> 简介

EISeg(Efficient Interactive Segmentation)基于飞桨开发的一个高效智能的交互式分割标注软件。它涵盖了通用、人像、遥感、医疗、视频等不同方向的高质量交互式分割模型。 另外，将EISeg获取到的标注应用到PaddleSeg提供的其他分割模型进行训练，便可得到定制化场景的高精度模型，打通分割任务从数据标注到模型训练及预测的全流程。

![4a9ed-a91y1](https://user-images.githubusercontent.com/71769312/141130688-e1529c27-aba8-4bf7-aad8-dda49808c5c7.gif)

## <img src="../docs/images/feature.png" width="20"/> 特性

  * 高效的半自动标注工具，已上线多个Top标注平台
  * 覆盖遥感、医疗、视频、3D医疗等众多垂类场景
  * 多平台兼容，简单易用，支持多类别标签管理

## <img src="../docs/images/chat.png" width="20"/> 技术交流

* 如果您对EISeg有任何问题和建议，欢迎在[GitHub Issues](https://github.com/PaddlePaddle/PaddleSeg/issues)提issue。
* 欢迎您加入EISeg微信群，和大家交流讨论、一起共建EISeg，而且可以**领取重磅学习大礼包🎁**。
  * 🔥 获取深度学习视频教程、图像分割论文合集
  * 🔥 获取PaddleSeg的历次直播视频，最新发版信息和直播动态
  * 🔥 获取PaddleSeg自建的人像分割数据集，整理的开源数据集
  * 🔥 获取PaddleSeg在垂类场景的预训练模型和应用合集，涵盖人像分割、交互式分割等等
  * 🔥 获取PaddleSeg的全流程产业实操范例，包括质检缺陷分割、抠图Matting、道路分割等等
<div align="center">
<img src="https://user-images.githubusercontent.com/30883834/216607378-94745b74-0ca1-4d39-a817-de94385496f1.jpg"  width = "200" />  
</div>

## <img src="../docs/images/teach.png" width="20"/> 使用教程
* [安装说明](docs/install.md)
* [图像分割标注](docs/image.md)
* [视频及3D医疗分割标注](docs/video.md)
* [遥感分割特色功能](docs/remote_sensing.md)
* [医疗分割特色功能](docs/medical.md)
* [图像检测标注](docs/det.md)
* [数据处理脚本文档](docs/tools.md)


## <img src="../docs/images/anli.png" width="20"/> 更新历史
- 2022.12.16  **1.1.0**：【1】新增检测标注能力，支持手工标注或使用PicoDet模型进行预标注； 【2】检测标注结果保存格式支持COCO, VOC及YOLO等多种格式；【3】分割新增LabelMe JSON保存格式。
- 2022.07.20  **1.0.0**：【1】新增交互式视频分割功能；【2】新增腹腔多器官3D标注模型【3】新增CT椎骨3D标注模型。
- 2022.04.10  **0.5.0**：【1】新增chest_xray模型；【2】新增MRSpineSeg模型；【3】新增铝板质检标注模型；【4】修复保存shp时可能坐标出错。
- 2021.11.16  **0.4.0**：【1】将动态图预测转换成静态图预测，单次点击速度提升十倍；【2】新增遥感图像标注功能，支持多光谱数据通道的选择；【3】支持大尺幅数据的切片（多宫格）处理；【4】新增医疗图像标注功能，支持读取dicom的数据格式，支持选择窗宽和窗位。
- 2021.09.16  **0.3.0**：【1】初步完成多边形编辑功能，支持对交互标注的结果进行编辑；【2】支持中/英界面；【3】支持保存为灰度/伪彩色标签和COCO格式；【4】界面拖动更加灵活；【5】标签栏可拖动，生成mask的覆盖顺序由上往下覆盖。
- 2021.07.07  **0.2.0**：新增contrib：EISeg，可实现人像和通用图像的快速交互式标注。




## 贡献者

- 感谢[Zhiliang Yu](https://github.com/yzl19940819), [Yizhou Chen](https://github.com/geoyee), [Lin Han](https://github.com/linhandev), [Jinrui Ding](https://github.com/Thudjr), [Yiakwy](https://github.com/yiakwy), [GT](https://github.com/GT-ZhangAcer), [Youssef Harby](https://github.com/Youssef-Harby), [Nick Nie](https://github.com/niecongchong) 等开发者及[RITM](https://github.com/saic-vul/ritm_interactive_segmentation)、[MiVOS](https://github.com/hkchengrex/MiVOS) 等算法支持。
- 感谢[LabelMe](https://github.com/wkentaro/labelme)和[LabelImg](https://github.com/tzutalin/labelImg)的标签设计。
- 感谢[Weibin Liao](https://github.com/MrBlankness)提供的ResNet50_DeeplabV3+预训练模型。
- 感谢[Junjie Guo](https://github.com/Guojunjie08)及[Jiajun Feng](https://github.com/richarddddd198)在椎骨模型上提供的技术支持。

## 学术引用

如果我们的项目在学术上帮助到你，请考虑以下引用：

```latex
@article{hao2022eiseg,
  title={EISeg: An Efficient Interactive Segmentation Tool based on PaddlePaddle},
  author={Hao, Yuying and Liu, Yi and Chen, Yizhou and Han, Lin and Peng, Juncai and Tang, Shiyu and Chen, Guowei and Wu, Zewu and Chen, Zeyu and Lai, Baohua},
  journal={arXiv e-prints},
  pages={arXiv--2210},
  year={2022}
}

@inproceedings{hao2021edgeflow,
  title={Edgeflow: Achieving practical interactive segmentation with edge-guided flow},
  author={Hao, Yuying and Liu, Yi and Wu, Zewu and Han, Lin and Chen, Yizhou and Chen, Guowei and Chu, Lutao and Tang, Shiyu and Yu, Zhiliang and Chen, Zeyu and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1551--1560},
  year={2021}
}
```

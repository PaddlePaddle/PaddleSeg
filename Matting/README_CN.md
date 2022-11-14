简体中文 | [English](README.md)

# Image Matting

## 目录
* [简介](#简介)
* [更新动态](#更新动态)
* [技术交流](#技术交流)
* [模型库](#模型库)
* [使用教程](#使用教程)
* [社区贡献](#社区贡献)
* [学术引用](#学术引用)


## 简介

Image Matting（精细化分割/影像去背/抠图）是指借由计算前景的颜色和透明度，将前景从影像中撷取出来的技术，可用于替换背景、影像合成、视觉特效，在电影工业中被广泛地使用。影像中的每个像素会有代表其前景透明度的值，称作阿法值（Alpha），一张影像中所有阿法值的集合称作阿法遮罩（Alpha Matte），将影像被遮罩所涵盖的部分取出即可完成前景的分离。


<p align="center">
<img src="https://user-images.githubusercontent.com/30919197/179751613-d26f2261-7bcf-4066-a0a4-4c818e7065f0.gif" width="100%" height="100%">
</p>

## 更新动态
### 最新动态
2022.11
【1】开源PP-MattingV2模型。对比MODNet模型推理速度提升44.6%， 误差平均相对减小17.91%。
【2】调整文档结构，完善模型库信息。
【3】[FastDeploy](https://github.com/PaddlePaddle/FastDeploy)支持PP-MattingV2, PP-Matting, PP-HumanMatting和MODNet模型。

### 更新历史
2022.07
【1】开源PP-Matting代码。
【2】新增ClosedFormMatting、KNNMatting、FastMatting、LearningBaseMatting和RandomWalksMatting传统机器学习算法。
【3】新增GCA模型。
【4】完善目录结构。
【5】支持指定指标进行评估。

2022.04
【1】新增PP-Matting模型。
【2】新增PP-HumanMatting高分辨人像抠图模型。
【3】新增Grad、Conn评估指标。
【4】新增前景评估功能，利用[ML](https://arxiv.org/pdf/2006.14970.pdf)算法在预测和背景替换时进行前景评估。
【5】新增GradientLoss和LaplacianLoss。
【6】新增RandomSharpen、RandomSharpen、RandomReJpeg、RSSN数据增强策略。

2021.11 Matting项目开源, 实现图像抠图功能。
【1】支持Matting模型：DIM， MODNet。
【2】支持模型导出及Python部署。
【3】支持背景替换功能。
【4】支持人像抠图Android部署

## 技术交流

* 如果大家有使用问题和功能建议, 可以通过[GitHub Issues](https://github.com/PaddlePaddle/PaddleSeg/issues)提issue。
* **欢迎大家加入PaddleSeg的微信用户群👫**（扫码填写问卷即可入群），和各界大佬交流学习，还可以**领取重磅大礼包🎁**
  * 🔥 获取PaddleSeg的历次直播视频，最新发版信息和直播动态
  * 🔥 获取PaddleSeg自建的人像分割数据集，整理的开源数据集
  * 🔥 获取PaddleSeg在垂类场景的预训练模型和应用合集，涵盖人像分割、交互式分割等等
  * 🔥 获取PaddleSeg的全流程产业实操范例，包括质检缺陷分割、抠图Matting、道路分割等等
<div align="center">
<img src="https://user-images.githubusercontent.com/48433081/174770518-e6b5319b-336f-45d9-9817-da12b1961fb1.jpg"  width = "200" />  
</div>

## [模型库](docs/model_zoo_cn.md)
开源多种场景高质量**人像抠图**模型，可根据实际应用场景直接部署应用，也可进行微调训练。具体信息请参考[model zoo](docs/model_zoo_cn.md)。

## 使用教程
* [在线体验](docs/online_demo_cn.md)
* [快速体验](docs/quick_start_cn.md)
* [全流程开发](docs/full_develop_cn.md)
* [人像抠图Android部署](deploy/human_matting_android_demo/README_CN.md)
* [数据集准备](docs/data_prepare_cn.md)
* AI Studio第三方教程
  * [PaddleSeg——Matting教程](https://aistudio.baidu.com/aistudio/projectdetail/3876411?contributionType=1)
  * [【PaddleSeg——Matting实践范例】PP-Matting图像抠图](https://aistudio.baidu.com/aistudio/projectdetail/5002963?contributionType=1)

## 社区贡献
* 感谢[钱彬(Qianbin)](https://github.com/qianbin1989228)等开发者的贡献。
* 感谢Jizhizi Li等提出的[GFM](https://arxiv.org/abs/2010.16188) Matting框架助力PP-Matting的算法研发。

## 学术引用
@article{chen2022pp,
  title={PP-Matting: High-Accuracy Natural Image Matting},
  author={Chen, Guowei and Liu, Yi and Wang, Jian and Peng, Juncai and Hao, Yuying and Chu, Lutao and Tang, Shiyu and Wu, Zewu and Chen, Zeyu and Yu, Zhiliang and others},
  journal={arXiv preprint arXiv:2204.09433},
  year={2022}
}

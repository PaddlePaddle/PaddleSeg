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

Image Matting（精细化分割/影像去背/抠图）是指借由计算前景的颜色和透明度，将前景从影像中撷取出来的技术，可用于替换背景、影像合成、视觉特效，在电影工业中被广泛地使用。
影像中的每个像素会有代表其前景透明度的值，称作阿法值（Alpha），一张影像中所有阿法值的集合称作阿法遮罩（Alpha Matte），将影像被遮罩所涵盖的部分取出即可完成前景的分离。


<p align="center">
<img src="https://user-images.githubusercontent.com/30919197/179751613-d26f2261-7bcf-4066-a0a4-4c818e7065f0.gif" width="100%" height="100%">
</p>

## 更新动态
* 2022.11
  * **开源自研轻量级抠图SOTA模型PP-MattingV2**。对比MODNet, PP-MattingV2推理速度提升44.6%， 误差平均相对减小17.91%。
  * 调整文档结构，完善模型库信息。
  * [FastDeploy](https://github.com/PaddlePaddle/FastDeploy)部署支持PP-MattingV2, PP-Matting, PP-HumanMatting和MODNet模型。
* 2022.07
  * 开源PP-Matting代码；新增ClosedFormMatting、KNNMatting、FastMatting、LearningBaseMatting和RandomWalksMatting传统机器学习算法；新增GCA模型。
  * 完善目录结构；支持指定指标进行评估。
* 2022.04
  * **开源自研高精度抠图SOTA模型PP-Matting**；新增PP-HumanMatting高分辨人像抠图模型。
  * 新增Grad、Conn评估指标；新增前景评估功能，利用[ML](https://arxiv.org/pdf/2006.14970.pdf)算法在预测和背景替换时进行前景评估。
  * 新增GradientLoss和LaplacianLoss；新增RandomSharpen、RandomSharpen、RandomReJpeg、RSSN数据增强策略。
* 2021.11
  * **Matting项目开源**, 实现图像抠图功能。
  * 支持Matting模型：DIM， MODNet；支持模型导出及Python部署；支持背景替换功能；支持人像抠图Android部署。

## 技术交流

* 如果大家有使用问题和功能建议, 可以通过[GitHub Issues](https://github.com/PaddlePaddle/PaddleSeg/issues)提issue。
* **欢迎加入PaddleSeg的微信用户群👫**（扫码填写简单问卷即可入群），大家可以和值班同学、各界大佬直接进行交流，还可以**领取30G重磅学习大礼包🎁**
  * 🔥 获取深度学习视频教程、图像分割论文合集
  * 🔥 获取PaddleSeg的历次直播视频，最新发版信息和直播动态
  * 🔥 获取PaddleSeg自建的人像分割数据集，整理的开源数据集
  * 🔥 获取PaddleSeg在垂类场景的预训练模型和应用合集，涵盖人像分割、交互式分割等等
  * 🔥 获取PaddleSeg的全流程产业实操范例，包括质检缺陷分割、抠图Matting、道路分割等等
<div align="center">
<img src="https://paddleseg.bj.bcebos.com/images/seg_qr_code.png"  width = "200" />  
</div>

## 模型库

针对高频应用场景 —— 人像抠图，我们训练并开源了**高质量人像抠图模型库**。根据实际应用场景，大家可以直接部署应用，也支持进行微调训练。

模型库中包括我们自研的高精度PP-Matting模型和轻量级PP-MattingV2模型。
- PP-Matting是PaddleSeg自研的高精度抠图模型，通过引导流设计实现语义引导下高分辨率图像抠图。追求更高精度，推荐使用该模型。
    且该模型提供了512和1024两个分辨率级别的预训练模型。
- PP-MattingV2是PaddleSeg自研的轻量级抠图SOTA模型，通过双层金字塔池化及空间注意力提取高级语义信息，并利用多级特征融合机制兼顾语义和细节的预测。
    对比MODNet模型推理速度提升44.6%， 误差平均相对减小17.91%。追求更高速度，推荐使用该模型。

| 模型 | SAD | MSE | Grad | Conn |Params(M) | FLOPs(G) | FPS | Config File | Checkpoint | Inference Model |
| - | - | -| - | - | - | - | -| - | - | - |
| PP-MattingV2-512   |40.59|0.0038|33.86|38.90| 8.95 | 7.51 | 98.89 |[cfg](../configs/ppmattingv2/ppmattingv2-stdc1-human_512.yml)| [model](https://paddleseg.bj.bcebos.com/matting/models/ppmattingv2-stdc1-human_512.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/ppmattingv2-stdc1-human_512.zip) |
| PP-Matting-512     |31.56|0.0022|31.80|30.13| 24.5 | 91.28 | 28.9 |[cfg](../configs/ppmatting/ppmatting-hrnet_w18-human_512.yml)| [model](https://paddleseg.bj.bcebos.com/matting/models/ppmatting-hrnet_w18-human_512.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/ppmatting-hrnet_w18-human_512.zip) |
| PP-Matting-1024    |66.22|0.0088|32.90|64.80| 24.5 | 91.28 | 13.4(1024X1024) |[cfg](../configs/ppmatting/ppmatting-hrnet_w18-human_1024.yml)| [model](https://paddleseg.bj.bcebos.com/matting/models/ppmatting-hrnet_w18-human_1024.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/ppmatting-hrnet_w18-human_1024.zip) |
| PP-HumanMatting    |53.15|0.0054|43.75|52.03| 63.9 | 135.8 (2048X2048)| 32.8(2048X2048)|[cfg](../configs/human_matting/human_matting-resnet34_vd.yml)| [model](https://paddleseg.bj.bcebos.com/matting/models/human_matting-resnet34_vd.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/pp-humanmatting-resnet34_vd.zip) |
| MODNet-MobileNetV2 |50.07|0.0053|35.55|48.37| 6.5 | 15.7 | 68.4 |[cfg](../configs/modnet/modnet-mobilenetv2.yml)| [model](https://paddleseg.bj.bcebos.com/matting/models/modnet-mobilenetv2.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/modnet-mobilenetv2.zip) |
| MODNet-ResNet50_vd |39.01|0.0038|32.29|37.38| 92.2 | 151.6 | 29.0 |[cfg](../configs/modnet/modnet-resnet50_vd.yml)| [model](https://paddleseg.bj.bcebos.com/matting/models/modnet-resnet50_vd.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/modnet-resnet50_vd.zip) |
| MODNet-HRNet_W18   |35.55|0.0035|31.73|34.07| 10.2 | 28.5 | 62.6 |[cfg](../configs/modnet/modnet-hrnet_w18.yml)| [model](https://paddleseg.bj.bcebos.com/matting/models/modnet-hrnet_w18.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/modnet-hrnet_w18.zip) |
| DIM-VGG16          |32.31|0.0233|28.89|31.45| 28.4 | 175.5| 30.4 |[cfg](../configs/dim/dim-vgg16.yml)| [model](https://paddleseg.bj.bcebos.com/matting/models/dim-vgg16.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/dim-vgg16.zip) |

**注意**：
* 指标计算数据集为PPM-100和AIM-500中的人像部分共同组成，共195张，[PPM-AIM-195](https://paddleseg.bj.bcebos.com/matting/datasets/PPM-AIM-195.zip)。
* FLOPs和FPS计算默认模型输入大小为(512, 512), GPU为Tesla V100 32G。FPS基于Paddle Inference预测库进行计算。
* DIM为trimap-based的抠图方法，指标只计算过度区域部分，对于没有提供trimap的情况下，默认将0<alpha<255的区域以25像素为半径进行膨胀腐蚀后作为过度区域。

## 使用教程
* [在线体验](docs/online_demo_cn.md)
* [快速体验](docs/quick_start_cn.md)
* [全流程开发](docs/full_develop_cn.md)
* [人像抠图Android部署](deploy/human_matting_android_demo/README_CN.md)
* [人像抠图.NET部署](https://gitee.com/raoyutian/PaddleSegSharp)
* [数据集准备](docs/data_prepare_cn.md)
* AI Studio第三方教程
  * [PaddleSeg的Matting教程](https://aistudio.baidu.com/aistudio/projectdetail/3876411?contributionType=1)
  * [PP-Matting图像抠图教程](https://aistudio.baidu.com/aistudio/projectdetail/5002963?contributionType=1)

## 社区贡献
* 感谢[钱彬(Qianbin)](https://github.com/qianbin1989228)等开发者的贡献。
* 感谢Jizhizi Li等提出的[GFM](https://arxiv.org/abs/2010.16188) Matting框架助力PP-Matting的算法研发。

## 学术引用
```
@article{chen2022pp,
  title={PP-Matting: High-Accuracy Natural Image Matting},
  author={Chen, Guowei and Liu, Yi and Wang, Jian and Peng, Juncai and Hao, Yuying and Chu, Lutao and Tang, Shiyu and Wu, Zewu and Chen, Zeyu and Yu, Zhiliang and others},
  journal={arXiv preprint arXiv:2204.09433},
  year={2022}
}
```

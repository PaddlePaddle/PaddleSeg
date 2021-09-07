简体中文 | [English](README.md)

# PaddleSeg

[![Build Status](https://travis-ci.org/PaddlePaddle/PaddleSeg.svg?branch=release/2.1)](https://travis-ci.org/PaddlePaddle/PaddleSeg)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleSeg.svg)](https://github.com/PaddlePaddle/PaddleSeg/releases)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)


## PaddleSeg重磅发布2.2版本，欢迎体验

* PaddleSeg团队在CVPR2021 AutoNUE语义分割赛道中获得冠军! 已发布[演讲报告](https://bj.bcebos.com/paddleseg/docs/autonue21_presentation_PaddleSeg.pdf)和[源代码](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.2/contrib/AutoNUE)。
* 发布了交互式分割的智能标注工具 [EISeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.2/contrib/EISeg)。极大的提升了标注效率；
* 开源了全景分割算法[Panoptic-DeepLab](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.2/contrib/PanopticDeepLab)丰富了模型种类；
* 全新升级了[人像分割](./contrib/PP-HumanSeg)功能，提供了web端超轻量模型部署方案

## PaddleSeg介绍
PaddleSeg是基于飞桨[PaddlePaddle](https://www.paddlepaddle.org.cn)开发的端到端图像分割开发套件，涵盖了**高精度**和**轻量级**等不同方向的大量高质量分割模型。通过模块化的设计，提供了**配置化驱动**和**API调用**两种应用方式，帮助开发者更便捷地完成从训练到部署的全流程图像分割应用。

* ### PaddleSeg提供了语义分割、交互式分割、全景分割、Matting四大图像分割能力。

<div align="center">
<img src="https://user-images.githubusercontent.com/53808988/130562440-1ea5cbf5-4caf-424c-a9a7-55d56b7d7776.gif"  width = "2000" />  
</div>




---------------

 * ### PaddleSeg被广泛地应用在自动驾驶、医疗、质检、巡检、娱乐等场景。

<div align="center">
<img src="https://user-images.githubusercontent.com/53808988/130562530-ae45c2cd-5dd7-48f0-a080-c0e843eea49d.gif"  width = "2000" />  
</div>

----------------
## 特性 <img src="./docs/images/feature.png" width="30"/>


* <img src="./docs/images/f1.png" width="20"/> **高精度模型**：基于百度自研的[半监督标签知识蒸馏方案（SSLD）](https://paddleclas.readthedocs.io/zh_CN/latest/advanced_tutorials/distillation/distillation.html#ssld)训练得到高精度骨干网络，结合前沿的分割技术，提供了50+的高质量预训练模型，效果优于其他开源实现。

* <img src="./docs/images/f2.png" width="20"/> **模块化设计**：支持20+主流 *分割网络* ，结合模块化设计的 *数据增强策略* 、*骨干网络*、*损失函数* 等不同组件，开发者可以基于实际应用场景出发，组装多样化的训练配置，满足不同性能和精度的要求。

* <img src="./docs/images/f3.png" width="20"/> **高性能**：支持多进程异步I/O、多卡并行训练、评估等加速策略，结合飞桨核心框架的显存优化功能，可大幅度减少分割模型的训练开销，让开发者更低成本、更高效地完成图像分割训练。
* :heart:**您可以前往  [完整PaddleSeg在线使用文档目录](https://paddleseg.readthedocs.io)  获得更详细的说明文档**:heart:
----------


## <img src="./docs/images/love.png" width="40"/> 直播课回放

✨直播课回放--全球冠军带你实现产业级图像分割✨  

* 学习链接：https://aistudio.baidu.com/aistudio/education/group/info/24590

* Day① 顶会冠军图像分割算法深度解密

* Day② 高精度人像分割算法及应用

* Day③ 交互式分割及破圈应用


## 技术交流 <img src="./docs/images/chat.png" width="30"/>

* 如果你发现任何PaddleSeg存在的问题或者是建议, 欢迎通过[GitHub Issues](https://github.com/PaddlePaddle/PaddleSeg/issues)给我们提issues。
* 欢迎加入PaddleSegQQ群
<div align="center">
<img src="./docs/images/QQ_chat.png"  width = "200" />  
</div>

## 模型说明  <img src="./docs/images/model.png" width="20"/>

[Model Zoo](./configs/)

<div align="center">
<img src="./docs/images/xingnengtu.png"    width = "700"/>  
</div>


## 使用教程 <img src="./docs/images/teach.png" width="30"/>

* [安装](./docs/install_cn.md)
* [全流程跑通PaddleSeg](./docs/whole_process_cn.md)
*  准备数据集
   * [标注数据的准备](./docs/data/marker/marker_cn.md)
   * [数据标注教程](./docs/data/transform/transform_cn.md)
   * [自定义数据集](./docs/data/custom/data_prepare_cn.md)

*  PaddleSeg二次开发教程
    * [配置文件详解](./docs/design/use/use_cn.md)
    * [如何创造自己的模型](./docs/design/create/add_new_model_cn.md)
* [模型训练](/docs/train/train_cn.md)
* [模型评估](./docs/evaluation/evaluate/evaluate.md)
* [预测与可视化](./docs/predict/predict_cn.md)
* [模型导出](./docs/export/export/model_export.md)

*  模型部署
    * [Inference](./docs/deployment/inference/inference.md)
    * [Lite](./docs/deployment/lite/lite.md)
    * [Serving](./docs/deployment/serving/serving.md)
    * [Web](./docs/deployment/web/web.md)
* 模型压缩
    * [量化](./docs/slim/quant/quant.md)
    * [裁剪](./docs/slim/prune/prune.md)
*  API使用教程
    * [API文档说明](./docs/apis/README_CN.md)
    * [API应用案例](./docs/api_example.md)
*  重要模块说明
    * [数据增强](./docs/module/data/data.md)
    * [Loss说明](./docs/module/loss/losses_cn.md)
    * [Tricks](./docs/module/tricks/tricks.md)
* 经典模型说明
    * [DeeplabV3](./docs/models/deeplabv3.md)
    * [UNet](./docs/models/unet.md)
    * [OCRNet](./docs/models/ocrnet.md)
    * [Fast-SCNN](./docs/models/fascnn.md)
* [提交PR说明](./docs/pr/pr/pr.md)
* [FAQ](./docs/faq/faq/faq_cn.md)

## 实践案例 <img src="./docs/images/anli.png" width="20"/>

- [人像分割](./contrib/PP-HumanSeg)
- [医疗图像](./docs/solution/medical/medical.md)
- [遥感分割](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.1/contrib/remote_sensing)
- [全景分割](./contrib/PanopticDeepLab)

## 代码贡献

- 非常感谢[jm12138](https://github.com/jm12138)贡献U<sup>2</sup>-Net模型。
- 非常感谢[zjhellofss](https://github.com/zjhellofss)（傅莘莘）贡献Attention U-Net模型，和Dice loss损失函数。
- 非常感谢[liuguoyu666](https://github.com/liguoyu666)贡献U-Net++模型。

## 学术引用 <img src="./docs/images/yinyong.png" width="30"/>

如果我们的项目在学术上帮助到你，请考虑以下引用：

```latex
@misc{liu2021paddleseg,
      title={PaddleSeg: A High-Efficient Development Toolkit for Image Segmentation},
      author={Yi Liu and Lutao Chu and Guowei Chen and Zewu Wu and Zeyu Chen and Baohua Lai and Yuying Hao},
      year={2021},
      eprint={2101.06175},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{paddleseg2019,
    title={PaddleSeg, End-to-end image segmentation kit based on PaddlePaddle},
    author={PaddlePaddle Authors},
    howpublished = {\url{https://github.com/PaddlePaddle/PaddleSeg}},
    year={2019}
}
```

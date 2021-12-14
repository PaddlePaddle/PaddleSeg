简体中文 | [English](README.md)

# PaddleSeg

[![Build Status](https://travis-ci.org/PaddlePaddle/PaddleSeg.svg?branch=release/2.1)](https://travis-ci.org/PaddlePaddle/PaddleSeg)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleSeg.svg)](https://github.com/PaddlePaddle/PaddleSeg/releases)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)

## 近期活动
PaddleSeg团队将举办主题为《产业图像分割应用与实战》的两日课向大家分析在**交互式智能标注工具**和**精细化分割Matting**方向的研究工作。
<div align="center">
<img src=https://user-images.githubusercontent.com/14087480/142155222-97b7b93d-3f2a-433a-be70-6b2a565e936b.png  width = "2000" />  
</div>

## PaddleSeg发布2.3版本，欢迎体验

* PaddleSeg团队发表交互式分割论文[EdgeFlow](https://arxiv.org/abs/2109.09406)，已在多个数据集实现SOTA性能，并升级了交互式分割工具[EISeg](./EISeg)。
* 开源两种[Matting](./contrib/Matting)算法，经典方法DIM，和实时性方法MODNet，实现精细化人像分割。
* 发布图像分割高阶功能，[模型蒸馏](./slim/distill)和[模型量化](./slim/quant)方案，进一步提升模型的部署效率。

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

----------

## 技术交流 <img src="./docs/images/chat.png" width="30"/>

* 如果你发现任何PaddleSeg存在的问题或者是建议, 欢迎通过[GitHub Issues](https://github.com/PaddlePaddle/PaddleSeg/issues)给我们提issues。
* 欢迎加入PaddleSegQQ群
<div align="center">
<img src="./docs/images/QQ_chat.png"  width = "200" />  
</div>

## 模型库总览  <img src="./docs/images/model.png" width="20"/>

更多信息参见[Model Zoo Overview](./docs/model_zoo_overview.md)

<div align="center">
<img src=https://user-images.githubusercontent.com/30695251/140323107-02ce9de4-c8f4-4f18-88b2-59bd0055a70b.png   />  
</div>


## 数据集

- [x] Cityscapes
- [x] Pascal VOC
- [x] ADE20K
- [x] Pascal Context
- [x] COCO stuff
- [x] CHASE_DB1
- [x] HRF
- [x] DRIVE
- [x] STARE
- [x] EG1800
- [x] SUPERVISELY

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
    * [提交PR说明](./docs/pr/pr/pr.md)
    * [模型PR规范](./docs/pr/pr/style_cn.md)
* [模型训练](/docs/train/train_cn.md)
* [模型评估](./docs/evaluation/evaluate/evaluate.md)
* [预测与可视化](./docs/predict/predict_cn.md)

* 模型导出
    * [导出预测模型](./docs/model_export.md)
    * [导出ONNX模型](./docs/model_export_onnx.md)

* 模型部署
    * [Paddle Inference部署(Python)](./docs/deployment/inference/python_inference.md)
    * [Paddle Inference部署(C++)](./docs/deployment/inference/cpp_inference.md)
    * [Paddle Lite部署](./docs/deployment/lite/lite.md)
    * [Paddle Serving部署](./docs/deployment/serving/serving.md)
    * [Paddle JS部署](./docs/deployment/web/web.md)
    * [推理Benchmark](./docs/deployment/inference/infer_benchmark.md)

* 模型压缩
    * [量化](./docs/slim/quant/quant.md)
    * [蒸馏](./docs/slim/distill/distill.md)
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
* [静态图版本](./docs/static/static_cn.md)
* [常见问题汇总](./docs/faq/faq/faq_cn.md)

## 实践案例 <img src="./docs/images/anli.png" width="20"/>

- [人像分割](./contrib/PP-HumanSeg)
- [Cityscapes打榜模型](./contrib/CityscapesSOTA)
- [CVPR冠军模型](./contrib/AutoNUE)
- [全景分割](./contrib/PanopticDeepLab)
- [交互式分割](./EISeg)
- [深度抠图](./contrib/Matting)

## 代码贡献

- 非常感谢[jm12138](https://github.com/jm12138)贡献U<sup>2</sup>-Net模型。
- 非常感谢[zjhellofss](https://github.com/zjhellofss)（傅莘莘）贡献Attention U-Net模型，和Dice loss损失函数。
- 非常感谢[liuguoyu666](https://github.com/liguoyu666)贡献U-Net++模型。
- 非常感谢[yazheng0307](https://github.com/yazheng0307) (刘正)贡献快速开始教程文档。
- 非常感谢[CuberrChen](https://github.com/CuberrChen)贡献STDC (rethink BiSeNet) PointRend，和 Detail Aggregate损失函数。
- 非常感谢[stuartchen1949](https://github.com/stuartchen1949)贡献 SegNet。
- 非常感谢[justld](https://github.com/justld)(郎督)贡献 ESPNet，HRNet_W48_Contrast 和 PixelContrastCrossEntropyLoss 损失函数。
- 非常感谢[Herman-Hu-saber](https://github.com/Herman-Hu-saber)(胡慧明)参与贡献 ESPNet。
- 非常感谢[zhangjin12138](https://github.com/zhangjin12138)贡献数据增强方法 RandomCenterCrop。


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

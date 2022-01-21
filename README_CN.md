简体中文 | [English](README.md)

<div align="center">

<p align="center">
  <img src="./docs/images/paddleseg_logo.png" align="middle" width = "500" />
</p>

**飞桨高性能图像分割开发套件，端到端地完成从训练到部署的全流程图像分割应用。**

[![Build Status](https://travis-ci.org/PaddlePaddle/PaddleSeg.svg?branch=release/2.1)](https://travis-ci.org/PaddlePaddle/PaddleSeg)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleSeg.svg)](https://github.com/PaddlePaddle/PaddleSeg/releases)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)
</div>



## 最新动态 <img src="./docs/images/seg_news_icon.png" width="40"/>

* [2022-01-20] PaddleSeg 2.4版本发布！详细发版信息请参考[Release Note](https://github.com/PaddlePaddle/PaddleSeg/releases)。
  * 发布智能标注工具[EISeg v0.4](./EISeg)版，采用静态图预测加速交互过程，并支持遥感和医疗领域智能标注。
  * 发表人像分割论文[PP-HumanSeg](./contrib/PP-HumanSeg/paper.md)，并开源连通性学习（SCL）方法和大规模视频会议数据集。
  * 开源发丝级人像抠图Matting算法[PP-HumanMatting](./contrib/Matting)，并提供安卓端教程。
  * 新增9个经典分割模型, 包括Transformer系列模型[SegMenter](configs/segmenter)，总模型数达到40个。
  * 提供分割模型[性能对比图](docs/model_zoo_overview_cn.md)，全面展示分割模型性能，方便模型选型。
* [2021-10-11] PaddleSeg 2.3版本发布交互式分割工具EISeg v0.3, 开源两种[Matting](./contrib/Matting)算法, 以及分割高阶功能[模型蒸馏](./slim/distill)和[模型量化](./slim/quant)方案。
* [2021-09-20] PaddleSeg团队发表交互式分割论文[EdgeFlow](https://arxiv.org/abs/2109.09406)，已在多个数据集实现SOTA性能。

## 简介
PaddleSeg是基于飞桨PaddlePaddle开发的端到端图像分割开发套件，涵盖了**高精度**和**轻量级**等不同方向的大量高质量分割模型。通过模块化的设计，提供了**配置化驱动**和**API调用**两种应用方式，帮助开发者更便捷地完成从训练到部署的全流程图像分割应用。

* #### 提供语义分割、交互式分割、全景分割、Matting四大图像分割能力。

<div align="center">
<img src="https://user-images.githubusercontent.com/53808988/130562440-1ea5cbf5-4caf-424c-a9a7-55d56b7d7776.gif"  width = "2000" />  
</div>




---------------

 * #### 广泛应用在自动驾驶、医疗、质检、巡检、娱乐等场景。

<div align="center">
<img src="https://user-images.githubusercontent.com/53808988/130562530-ae45c2cd-5dd7-48f0-a080-c0e843eea49d.gif"  width = "2000" />  
</div>

----------------
## 特性 <img src="./docs/images/feature.png" width="30"/>


* <img src="./docs/images/f1.png" width="20"/> **高精度模型**：基于半监督标签知识蒸馏方案([SSLD](https://paddleclas.readthedocs.io/zh_CN/latest/advanced_tutorials/distillation/distillation.html#ssld))训练得到高精度骨干网络，结合前沿的分割技术，提供了80+的高质量预训练模型，效果优于其他开源实现。

* <img src="./docs/images/f2.png" width="20"/> **模块化设计**：支持40+主流 *分割网络* ，结合模块化设计的 *数据增强策略* 、*骨干网络*、*损失函数* 等不同组件，开发者可以基于实际应用场景出发，组装多样化的训练配置，满足不同性能和精度的要求。

* <img src="./docs/images/f3.png" width="20"/> **高性能**：支持多进程异步I/O、多卡并行训练、评估等加速策略，结合飞桨核心框架的显存优化功能，可大幅度减少分割模型的训练开销，让开发者更低成本、更高效地完成图像分割训练。

----------

## 技术交流 <img src="./docs/images/chat.png" width="30"/>

* 如果你发现任何PaddleSeg存在的问题或者是建议, 欢迎通过[GitHub Issues](https://github.com/PaddlePaddle/PaddleSeg/issues)给我们提issues。
* 欢迎加入PaddleSeg QQ群
<div align="center">
<img src="./docs/images/QQ_chat.png"  width = "200" />  
</div>

## 产品矩阵 <img src="./docs/images/model.png" width="20"/>

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>分割模型</b>
      </td>
      <td colspan="2">
        <b>分割组件</b>
      </td>
      <td>
        <b>实践案例</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
            <li>ANN</li>
            <li>BiSeNetV2</li>
            <li>DANet</li>
            <li>DeepLabV3</li>
            <li>DeepLabV3P</li>
            <li>Fast-SCNN</li>
            <li>HRNet-FCN</li>
            <li>GCNet</li>
            <li>GSCNN</li>
            <li>HarDNet</li>
            <li>OCRNet</li>
            <li>PSPNet</li>
            <li>U-Net</li>
            <li>U<sup>2</sup>-Net</li>
            <li>Att U-Net</li>
            <li>U-Net++</li>
            <li>U-Net3+</li>
            <li>DecoupledSeg</li>
            <li>EMANet</li>
            <li>ISANet</li>
            <li>DNLNet</li>
            <li>SFNet</li>
            <li>PP-HumanSeg</li>
            <li>PortraitNet</li>
            <li>STDC</li>
            <li>GINet</li>
            <li>PointRend</li>
            <li>SegNet</li>
            <li>ESPNetV2</li>
            <li>HRNet-Contrast</li>
            <li>DMNet</li>
            <li>ESPNetV1</li>
            <li>ENCNet</li>
            <li>PFPNNet</li>
            <li>FastFCN</li>
            <li>BiSeNetV1</li>
            <li>SETR</li>
            <li>MLA Transformer</li>
            <li>SegFormer</li>
            <li>SegMenter</li>
      </td>
      <td>
        <b>骨干网络</b><br>
          <ul>
            <li>HRNet</li>
            <li>MobileNetV2</li>
            <li>MobileNetV3</li>
            <li>ResNet</li>
            <li>STDCNet</li>
            <li>XCeption</li>
            <li>VIT</li>
            <li>MixVIT</li>
            <li>Swin Transformer</li>
          </ul>  
        <b>损失函数</b><br>
          <ul>
            <li>Cross Entropy</li>
            <li>Binary CE</li>
            <li>Bootstrapped CE</li>
            <li>Point CE</li>
            <li>OHEM CE</li>
            <li>Pixel Contrast CE</li>
            <li>Focal</li>
            <li>Dice</li>
            <li>RMI</li>
            <li>KL</li>
            <li>L1</li>
            <li>Lovasz</li>
            <li>MSE</li>
            <li>Edge Attention</li>
            <li>Relax Boundary</li>
            <li>Connectivity</li>
          </ul>
        <b>评估指标</b><br>
          <ul>
            <li>mIoU</li>
            <li>Accuracy</li>
            <li>Kappa</li>
            <li>Dice</li>
            <li>AUC_ROC</li>
          </ul>  
      </td>
      <td>
        <b>支持数据集</b><br>
          <ul>
            <li>Cityscapes</li>
            <li>Pascal VOC</li>
            <li>ADE20K</li>  
            <li>Pascal Context</li>  
            <li>COCO Stuff</li>
            <li>SUPERVISELY</li>
            <li>EG1800</li>
            <li>CHASE_DB1</li>
            <li>HRF</li>
            <li>DRIVE</li>
            <li>STARE</li>
            <li>PP-HumanSeg14K</li>
          </ul>
        <b>数据增强</b><br>
        <ul>
          <li>Flipping</li>  
          <li>Resize</li>  
          <li>ResizeByLong</li>
          <li>ResizeByShort</li>
          <li>LimitLong</li>  
          <li>ResizeRangeScaling</li>  
          <li>ResizeStepScaling</li>
          <li>Normalize</li>
          <li>Padding</li>
          <li>PaddingByAspectRatio</li>
          <li>RandomPaddingCrop</li>  
          <li>RandomCenterCrop</li>
          <li>ScalePadding</li>
          <li>RandomNoise</li>  
          <li>RandomBlur</li>  
          <li>RandomRotation</li>  
          <li>RandomScaleAspect</li>  
          <li>RandomDistort</li>  
          <li>RandomAffine</li>  
        </ul>  
      </td>
      <td>
        <b>交互式分割</b><br>
          <ul>
            <li>EISeg</li>
            <li>RITM</li>
            <li>EdgeFlow</li>
           </ul>
       <b>图像抠图</b><br>
        <ul>
            <li>DIM</li>
            <li>MODNet</li>
            <li>PP-HumanMatting</li>
        </ul>
        <b>人像分割</b><br>
        <ul>
            <li>PP-HumanSeg</li>
        </ul>
        <b>Cityscapes打榜模型</b><br>
        <ul>
            <li>HMSA</li>
        </ul>
        <b>全景分割</b><br>
          <ul>
            <li>Panoptic-DeepLab</li>
          </ul>
        <b>CVPR冠军模型</b><br>
        <ul>
            <li>MLA Transformer</li>
        </ul>
        <b>领域自适应</b><br>
        <ul>
            <li>PixMatch</li>
        </ul>
      </td>  
    </tr>


</td>
    </tr>
  </tbody>
</table>

## 模型库总览  <img src="./docs/images/model.png" width="20"/>

模型结构和骨干网络的代表模型在Cityscapes数据集mIoU和FLOPs对比图。请参见[Model Zoo Overview](./docs/model_zoo_overview_cn.md)了解更多模型信息以及对比图。

<div align="center">
<img src=https://user-images.githubusercontent.com/30695251/140323107-02ce9de4-c8f4-4f18-88b2-59bd0055a70b.png   />  
</div>


## 使用教程 <img src="./docs/images/teach.png" width="30"/>

* [安装](./docs/install_cn.md)
* [全流程跑通PaddleSeg](./docs/whole_process_cn.md)
*  准备数据集
   * [标注数据的准备](./docs/data/marker/marker_cn.md)
   * [数据标注教程](./docs/data/transform/transform_cn.md)
   * [自定义数据集](./docs/data/custom/data_prepare_cn.md)

* [模型训练](/docs/train/train_cn.md)
* [模型评估](./docs/evaluation/evaluate/evaluate_cn.md)
* [预测与可视化](./docs/predict/predict_cn.md)

* 模型导出
    * [导出预测模型](./docs/model_export_cn.md)
    * [导出ONNX模型](./docs/model_export_onnx_cn.md)

* 模型部署
    * [Paddle Inference部署(Python)](./docs/deployment/inference/python_inference_cn.md)
    * [Paddle Inference部署(C++)](./docs/deployment/inference/cpp_inference_cn.md)
    * [Paddle Lite部署](./docs/deployment/lite/lite_cn.md)
    * [Paddle Serving部署](./docs/deployment/serving/serving.md)
    * [Paddle JS部署](./docs/deployment/web/web_cn.md)
    * [推理Benchmark](./docs/deployment/inference/infer_benchmark_cn.md)

* 模型压缩
    * [量化](./docs/slim/quant/quant_cn.md)
    * [蒸馏](./docs/slim/distill/distill_cn.md)
    * [裁剪](./docs/slim/prune/prune_cn.md)

*  API使用教程
    * [API文档说明](./docs/apis/README_CN.md)
    * [API应用案例](./docs/api_example_cn.md)
*  重要模块说明
    * [数据增强](./docs/module/data/data_cn.md)
    * [Loss说明](./docs/module/loss/losses_cn.md)
*  二次开发教程
    * [配置文件详解](./docs/design/use/use_cn.md)
    * [如何创造自己的模型](./docs/design/create/add_new_model_cn.md)
*  模型贡献
    * [提交PR说明](./docs/pr/pr/pr_cn.md)
    * [模型PR规范](./docs/pr/pr/style_cn.md)

* [静态图版本](./docs/static/static_cn.md)
* [技术交流](#技术交流)
* [常见问题汇总](./docs/faq/faq/faq_cn.md)

## 实践案例 <img src="./docs/images/anli.png" width="20"/>

- [交互式分割](./EISeg)
- [图像抠图](./contrib/Matting)
- [人像分割](./contrib/PP-HumanSeg)
- [Cityscapes打榜模型](./contrib/CityscapesSOTA)
- [全景分割](./contrib/PanopticDeepLab)
- [CVPR冠军模型](./contrib/AutoNUE)
- [领域自适应](./contrib/DomainAdaptation)

## 第三方教程推荐

* [图像分割套件PaddleSeg全面解析系列](https://blog.csdn.net/txyugood/article/details/111029854)
* [PaddleSeg学习笔记: 人像分割 HumanSeg](https://blog.csdn.net/libo1004/article/details/118809026)

## 许可证书
本项目的发布受Apache 2.0 license许可认证。

## 社区贡献

- 非常感谢[jm12138](https://github.com/jm12138)贡献U<sup>2</sup>-Net模型。
- 非常感谢[zjhellofss](https://github.com/zjhellofss)（傅莘莘）贡献Attention U-Net模型，和Dice loss损失函数。
- 非常感谢[liuguoyu666](https://github.com/liguoyu666)贡献U-Net++模型。
- 非常感谢[yazheng0307](https://github.com/yazheng0307) (刘正)贡献快速开始教程文档。
- 非常感谢[CuberrChen](https://github.com/CuberrChen)贡献STDC (rethink BiSeNet) PointRend，和 Detail Aggregate损失函数。
- 非常感谢[stuartchen1949](https://github.com/stuartchen1949)贡献 SegNet。
- 非常感谢[justld](https://github.com/justld)(郎督)贡献 ESPNetV2, DMNet, ENCNet, HRNet_W48_Contrast, BiSeNetV1, FastFCN, SECrossEntropyLoss 和PixelContrastCrossEntropyLoss。
- 非常感谢[Herman-Hu-saber](https://github.com/Herman-Hu-saber)(胡慧明)参与贡献 ESPNetV2。
- 非常感谢[zhangjin12138](https://github.com/zhangjin12138)贡献数据增强方法 RandomCenterCrop。
- 非常感谢[simuler](https://github.com/simuler) 贡献 ESPNetV1。
- 非常感谢[ETTR123](https://github.com/ETTR123)(张恺) 贡献 PFPNNet。


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

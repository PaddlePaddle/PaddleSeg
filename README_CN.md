简体中文 | [English](README.md)

<div align="center">

<p align="center">
  <img src="./docs/images/paddleseg_logo.png" align="middle" width = "500" />
</p>

**强大易用的飞桨图像分割开发套件，端到端完成从训练到部署的全流程图像分割应用。**

[![Build Status](https://travis-ci.org/PaddlePaddle/PaddleSeg.svg?branch=release/2.1)](https://travis-ci.org/PaddlePaddle/PaddleSeg)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleSeg.svg)](https://github.com/PaddlePaddle/PaddleSeg/releases)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)
</div>



## <img src="./docs/images/seg_news_icon.png" width="20"/> 最新动态

* [2022-04-20] :fire: PaddleSeg 2.5版本发布！详细发版信息请参考[Release Note](https://github.com/PaddlePaddle/PaddleSeg/releases)。
  * 发布超轻量级语义分割模型[PP-LiteSeg](./configs/pp_liteseg)以及[技术报告](https://arxiv.org/pdf/2204.02681.pdf)，实现精度和速度的最佳平衡。
  * 发布高精度trimap-free抠图模型[PP-Matting](./Matting)以及[技术报告](https://arxiv.org/abs/2204.09433)，在Composition-1K和Distinctions-646上实现SOTA指标。
  * 发布3D医疗影像开发套件[MedicalSeg](./contrib/MedicalSeg)，支持数据预处理、模型训练、模型部署等全流程开发，并提供肺部、椎骨数据上的高精度分割模型。
  * 升级智能标注工具[EISeg v0.5](./EISeg)版，新增X-Ray胸腔标注、MRI椎骨标注、铝板瑕疵标注。
  * 新增5个经典分割模型, 包括多个版本的PP-LiteSeg，总模型数达到45个。
* [2022-01-20] PaddleSeg 2.4版本发布交互式分割工具EISeg v0.4，超轻量级人像分割方案[PP-HumanSeg](./contrib/PP-HumanSeg)，以及大规模视频会议数据集[PP-HumanSeg14K](./contrib/PP-HumanSeg/paper.md#pp-humanseg14k-a-large-scale-teleconferencing-video-dataset)。
* [2021-10-11] PaddleSeg 2.3版本发布交互式分割工具EISeg v0.3，开源两种[Matting](./contrib/Matting)算法，以及分割高阶功能[模型蒸馏](./slim/distill)和[模型量化](./slim/quant)方案。



## <img src="https://user-images.githubusercontent.com/48054808/157795569-9fc77c85-732f-4870-9be0-99a7fe2cff27.png" width="20"/> 简介

**PaddleSeg**是基于飞桨PaddlePaddle的端到端图像分割套件，内置**40+模型算法**及**140+预训练模型**，支持**配置化驱动**和**API调用**开发方式，打通数据标注、模型开发、训练、压缩、部署的**全流程**，提供**语义分割、交互式分割、Matting、全景分割**四大分割能力，助力算法在工业、自动驾驶、医疗、娱乐等场景落地应用。

<div align="center">
<img src="https://user-images.githubusercontent.com/53808988/130562440-1ea5cbf5-4caf-424c-a9a7-55d56b7d7776.gif"  width = "800" />  
</div>
<div align="center">
<img src="https://user-images.githubusercontent.com/53808988/130562530-ae45c2cd-5dd7-48f0-a080-c0e843eea49d.gif"  width = "800" />  
</div>

## <img src="./docs/images/feature.png" width="20"/> 特性

* **高精度**：跟踪学术界的前沿分割技术，结合半监督标签知识蒸馏方案([SSLD](https://paddleclas.readthedocs.io/zh_CN/latest/advanced_tutorials/distillation/distillation.html#ssld))训练的骨干网络，提供40+主流分割网络、140+的高质量预训练模型，效果优于其他开源实现。

* **高性能**：使用多进程异步I/O、多卡并行训练、评估等加速策略，结合飞桨核心框架的显存优化功能，大幅度减少分割模型的训练开销，让开发者更低成本、更高效地完成图像分割训练。

* **模块化**：源于模块化设计思想，解耦数据准备、分割模型、骨干网络、损失函数等不同组件，开发者可以基于实际应用场景出发，组装多样化的配置，满足不同性能和精度的要求。

* **全流程**：打通数据标注、模型开发、模型训练、模型压缩、模型部署全流程，经过业务落地的验证，让开发者完成一站式开发工作。

<div align="center">
<img src="https://user-images.githubusercontent.com/14087480/176379006-7f330e00-b6b0-480e-9df8-8fd1090da4cf.png" width = "800" />  
</div>

## <img src="./docs/images/chat.png" width="20"/> 技术交流

* 如果大家有使用问题、产品建议、功能需求, 可以通过[GitHub Issues](https://github.com/PaddlePaddle/PaddleSeg/issues)提issues。
* 欢迎大家扫码加入PaddleSeg微信群，和小伙伴们一起交流学习。

<div align="center">
<img src="https://user-images.githubusercontent.com/48433081/163670184-43cfb3ae-2047-4ba3-8dae-6c02090dd177.png"  width = "200" />  
</div>

## <img src="./docs/images/model.png" width="20"/> 产品矩阵

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>模型</b>
      </td>
      <td colspan="2">
        <b>组件</b>
      </td>
      <td>
        <b>特色案例</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        <details><summary><b>语义分割模型</b></summary>
          <ul>
            <li>PP-LiteSeg :star:</li>
            <li>DeepLabV3P :star: </li>
            <li>OCRNet :star: </li>
            <li>MobileSeg :star: </li>
            <li>ANN</li>
            <li>Att U-Net</li>
            <li>BiSeNetV1</li>
            <li>BiSeNetV2</li>
            <li>CCNet</li>
            <li>DANet</li>
            <li>DDRNet</li>
            <li>DecoupledSeg</li>
            <li>DeepLabV3</li>
            <li>DMNet</li>
            <li>DNLNet</li>
            <li>ESPNetV1</li>
            <li>ESPNetV2</li>
            <li>EMANet</li>
            <li>ENet</li>
            <li>ENCNet</li>
            <li>FastFCN</li>
            <li>Fast-SCNN</li>
            <li>GCNet</li>
            <li>GSCNN</li>
            <li>GINet</li>
            <li>GloRe</li>
            <li>HarDNet</li>
            <li>HRNet-FCN</li>
            <li>HRNet-Contrast</li>
            <li>ISANet</li>
            <li>MLA Transformer</li>
            <li>PSPNet</li>
            <li>PP-HumanSeg</li>
            <li>PortraitNet</li>
            <li>PointRend</li>
            <li>PFPNNet</li>
            <li>SegNet</li>
            <li>STDCSeg</li>
            <li>SFNet</li>
            <li>SETR</li>
            <li>SegFormer</li>
            <li>SegMenter</li>
            <li>U-Net</li>
            <li>U<sup>2</sup>-Net</li>
            <li>U-Net++</li>
            <li>U-Net3+</li>
          </ul>
        </details>
        <details><summary><b>交互式分割模型</b></summary>
          <ul>
            <li>EISeg</li>
            <li>RITM</li>
            <li>EdgeFlow</li>
          </ul>
        </details>
        <details><summary><b>图像抠图模型</b></summary>
          <ul>
              <li>PP-Matting</li>
              <li>DIM</li>
              <li>MODNet</li>
              <li>PP-HumanMatting</li>
          </ul>
        </details>
        <details><summary><b>全景分割</b></summary>
          <ul>
            <li>Panoptic-DeepLab</li>
          </ul>
        </details>
      </td>
      <td>
        <details><summary><b>骨干网络</b></summary>
          <ul>
            <li>HRNet</li>
            <li>ResNet</li>
            <li>STDCNet</li>
            <li>MobileNetV2</li>
            <li>MobileNetV3</li>
            <li>ShuffleNetV2</li>
            <li>GhostNet</li>
            <li>LiteHRNet</li>
            <li>XCeption</li>
            <li>VIT</li>
            <li>MixVIT</li>
            <li>Swin Transformer</li>
          </ul>
        </details>
        <details><summary><b>损失函数</b></summary>
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
            <li>MultiClassFocal</li>
          </ul>
        </details>
        <details><summary><b>评估指标</b></summary>
          <ul>
            <li>mIoU</li>
            <li>Accuracy</li>
            <li>Kappa</li>
            <li>Dice</li>
            <li>AUC_ROC</li>
          </ul>  
        </details>
      </td>
      <td>
        <details><summary><b>支持数据集</b></summary>
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
        </details>
        <details><summary><b>数据增强</b></summary>
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
        </details>
      </td>
      <td>
        <details><summary><b>人像分割模型</b></summary>
          <ul>
              <li>PP-HumanSeg</li>
          </ul>
        </details>
        <details><summary><b>3D医疗分割模型</b></summary>
          <ul>
            <li>VNet</li>
          </ul>
        </details>
        <details><summary><b>Cityscapes打榜模型</b></summary>
          <ul>
              <li>HMSA</li>
          </ul>
        </details>
        <details><summary><b>CVPR冠军模型</b></summary>
          <ul>
              <li>MLA Transformer</li>
          </ul>
        </details>
        <details><summary><b>领域自适应</b></summary>
          <ul>
              <li>PixMatch</li>
          </ul>
        </details>
      </td>  
    </tr>
</td>
    </tr>
  </tbody>
</table>

## <img src="https://user-images.githubusercontent.com/48054808/157801371-9a9a8c65-1690-4123-985a-e0559a7f9494.png" width="20"/> 产业级分割模型库

<details>
<summary><b>高精度语义分割模型</b></summary>

#### 高精度模型，分割mIoU高、推理算量大，适合部署在服务器端GPU和Jetson等设备。

| 模型名称  | 骨干网络   | Cityscapes精度mIoU(%) |  V100 TRT推理速度(FPS)  |  配置文件 |
|:-------- |:--------:|:---------------------:|:---------------------:|:--------:|
| FCN            | HRNet_W18        | 78.97                 | 24.43     | [yml](./configs/fcn/)         |
| FCN            | HRNet_W48        | 80.70                 | 10.16     | [yml](./configs/fcn/)         |
| DeepLabV3      | ResNet50_OS8     | 79.90                 | 4.56      | [yml](./configs/deeplabv3/)   |
| DeepLabV3      | ResNet101_OS8    | 80.85                 | 3.2       | [yml](./configs/deeplabv3/)   |
| DeepLabV3      | ResNet50_OS8     | 80.36                 | 6.58      | [yml](./configs/deeplabv3p/)  |
| DeepLabV3      | ResNet101_OS8    | 81.10                 | *3.94*    | [yml](./configs/deeplabv3p/)  |
| **OCRNet**     | HRNet_w18        | 80.67                 | 13.26     | [yml](./configs/ocrnet/)      |
| OCRNet         | HRNet_w48        | 82.15                 | 6.17      | [yml](./configs/ocrnet/)      |
| CCNet          | ResNet101_OS8    | 80.95                 | 3.24      | [yml](./configs/ccnet/)       |


测试条件：
* V100上测速条件：针对Nvidia GPU V100，使用PaddleInference预测库的Python API，开启TensorRT加速，数据类型是FP32，输入图像维度是1x3x1024x2048。

</details>


<details>
<summary><b>轻量级语义分割模型</b></summary>

#### 轻量级模型，分割mIoU中等、推理算量中等，可以部署在服务器端GPU、服务器端X86 CPU和移动端ARM CPU。

| 模型名称  | 骨干网络   | Cityscapes精度mIoU(%) |  V100 TRT推理速度(FPS) | 骁龙855推理速度(FPS) |  配置文件 |
|:-------- |:--------:|:---------------------:|:---------------------:|:-----------------:|:--------:|
| **PP-LiteSeg**      | STDC1         | 77.04               | 69.82           | 17.22       | [yml](./configs/pp_liteseg/)  |
| **PP-LiteSeg**      | STDC2         | 79.04               | 54.53           | 11.75       | [yml](./configs/pp_liteseg/)  |
| BiSeNetV1           | -             | 75.19               | 14.67           | 1.53      |[yml](./configs/bisenetv1/)  |
| BiSeNetV2           | -             | 73.19               | 61.83           | 13.67       |[yml](./configs/bisenet/)  |
| STDCSeg             | STDC1         | 74.74               | 62.24           | 14.51       |[yml](./configs/stdcseg/)  |
| STDCSeg             | STDC2         | 77.60               | 51.15           | 10.95       |[yml](./configs/stdcseg/)  |
| DDRNet_23           | -             | 79.85               | 42.64           | 7.68      |[yml](./configs/ddrnet/)  |
| HarDNet             | -             | 79.03               | 30.3            | 5.44      |[yml](./configs/hardnet/)  |
| SFNet               | ResNet18_OS8  |  78.72              | *10.72*         |   -         | [yml](./configs/sfnet/)  |

测试条件：
* V100上测速条件：针对Nvidia GPU V100，使用PaddleInference预测库的Python API，开启TensorRT加速，数据类型是FP32，输入图像维度是1x3x1024x2048。
* 骁龙855上测速条件：针对小米9手机，使用PaddleLite预测库的CPP API，ARMV8编译，单线程，输入图像维度是1x3x256x256。

</details>


<details>
<summary><b>超轻量级语义分割模型</b></summary>

#### 超轻量级模型，分割mIoU一般、推理算量低，适合部署在服务器端X86 CPU和移动端ARM CPU。

| 模型名称  | 骨干网络   | Cityscapes精度mIoU(%) |  V100 TRT推理速度(FPS)  | 骁龙855推理速度(FPS)|  配置文件 |
|:-------- |:--------:|:---------------------:|:---------------------:|:-----------------:|:--------:|
| MobileSeg      | MobileNetV2              | 73.94                 | 67.57          | 27.01   | [yml](./configs/mobileseg/)  |
| **MobileSeg**  | MobileNetV3              | 73.47                 | 67.39          | 32.90   | [yml](./configs/mobileseg/)  |
| MobileSeg      | Lite_HRNet_18            | 70.75                 | *10.5*         | 13.05   | [yml](./configs/mobileseg/)  |
| MobileSeg      | ShuffleNetV2_x1_0        | 69.46                 | *37.09*        | 39.61  | [yml](./configs/mobileseg/)  |
| MobileSeg      | GhostNet_x1_0            | 71.88                 | *35.58*        | 38.74  | [yml](./configs/mobileseg/)  |

测试条件：
* V100上测速条件：针对Nvidia GPU V100，使用PaddleInference预测库的Python API，开启TensorRT加速，数据类型是FP32，输入图像维度是1x3x1024x2048。
* 骁龙855上测速条件：针对小米9手机，使用PaddleLite预测库的CPP API，ARMV8编译，单线程，输入图像维度是1x3x256x256。

</details>

## <img src="./docs/images/teach.png" width="20"/> 使用教程

### 基础教程

* [安装说明](./docs/install_cn.md)
* [快速体验](./docs/quick_start_cn.md)
* 准备数据
   * [标注数据的准备](./docs/data/marker/marker_cn.md)
   * [数据标注教程](./docs/data/transform/transform_cn.md)
   * [自定义数据集](./docs/data/custom/data_prepare_cn.md)
* [准备配置文件](./docs/prepare_cfg_cn.md)

* [模型训练](/docs/train/train_cn.md)
* [模型评估](./docs/evaluation/evaluate/evaluate_cn.md)
* [模型预测](./docs/predict/predict_cn.md)

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
### 高阶教程

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

## <img src="./docs/images/anli.png" width="20"/> 实践案例

- [交互式分割](./EISeg)
- [图像抠图](./Matting)
- [人像分割](./contrib/PP-HumanSeg)
- [3D医疗分割](./contrib/MedicalSeg)
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
- 非常感谢[justld](https://github.com/justld)(郎督)贡献 UPerNet, DDRNet, CCNet, ESPNetV2, DMNet, ENCNet, HRNet_W48_Contrast, BiSeNetV1, FastFCN, SECrossEntropyLoss 和PixelContrastCrossEntropyLoss。
- 非常感谢[Herman-Hu-saber](https://github.com/Herman-Hu-saber)(胡慧明)参与贡献 ESPNetV2。
- 非常感谢[zhangjin12138](https://github.com/zhangjin12138)贡献数据增强方法 RandomCenterCrop。
- 非常感谢[simuler](https://github.com/simuler) 贡献 ESPNetV1。
- 非常感谢[ETTR123](https://github.com/ETTR123)(张恺) 贡献 ENet，PFPNNet。


## <img src="./docs/images/yinyong.png" width="20"/> 学术引用

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

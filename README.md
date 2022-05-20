简体中文 | [English](README_EN.md)

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



## News <img src="./docs/images/seg_news_icon.png" width="40"/>
<ul class="nobull">
  <li>[2022-04-20] :fire: PaddleSeg v2.5 is released! More details in <a href="https://github.com/PaddlePaddle/PaddleSeg/releases">Release Notes</a>.</li>
    <ul>
        <li>Release <a href="./configs/pp_liteseg">PP-LiteSeg</a>, a real-time semantic segmentation model. It achieves SOTA trade-off between segmentation accuracy and inference speed. [<a href="https://arxiv.org/pdf/2204.02681.pdf">techical report</a>]</li>
        <li>Release <a href="./Matting">PP-Matting</a>, a trimap-free image matting model for extremely fine-grained segmentation. It achieves SOTA performance on Composition-1k and Distinctions-646. [<a href="https://arxiv.org/abs/2204.09433">techical report</a>]</li>
        <li>Release <a href="./contrib/MedicalSeg">MedicalSeg</a>, a newly easy-to-use toolkit for 3D medical image segmentation. It supports the whole process including data preprocessing, model training, and model deployment, and provides the high-accuracy models on lung and spine segmentation.
        <li>Upgrade the interactive annotation tool <a href="./EISeg">EISeg v0.5</a> with supporting new areas in chest X-Ray, MRI spine, and defect inspection.</li>
        <li>Add 5 semantic segmentatioin models, including variants of PP-LiteSeg.</li>
    </ul>
 <li>[2022-01-20] We release PaddleSeg v2.4 with EISeg v0.4, and <a href="./contrib/PP-HumanSeg">PP-HumanSeg</a> including open-sourced dataset <a href="./contrib/PP-HumanSeg/paper.md#pp-humanseg14k-a-large-scale-teleconferencing-video-dataset">PP-HumanSeg14K</a>. </li>
 <li>[2021-10-11] We released PaddleSeg v2.3 with the improved interactive segmentation tool EISeg v0.3, two matting algorithms, and segmentation model compression.</li>


* [2022-04-20] :fire: PaddleSeg 2.5版本发布！详细发版信息请参考[Release Note](https://github.com/PaddlePaddle/PaddleSeg/releases)。
  * 发布超轻量级语义分割模型[PP-LiteSeg](./configs/pp_liteseg)以及[技术报告](https://arxiv.org/pdf/2204.02681.pdf)，实现精度和速度的最佳平衡。
  * 发布高精度trimap-free抠图模型[PP-Matting](./Matting)以及[技术报告](https://arxiv.org/abs/2204.09433)，在Composition-1K和Distinctions-646上实现SOTA指标。
  * 发布3D医疗影像开发套件[MedicalSeg](./contrib/MedicalSeg)，支持数据预处理、模型训练、模型部署等全流程开发，并提供肺部、椎骨数据上的高精度分割模型。
  * 升级智能标注工具[EISeg v0.5](./EISeg)版，新增X-Ray胸腔标注、MRI椎骨标注、铝板瑕疵标注。
  * 新增5个经典分割模型, 包括多个版本的PP-LiteSeg，总模型数达到45个。
* [2022-01-20] PaddleSeg 2.4版本发布交互式分割工具EISeg v0.4，超轻量级人像分割方案[PP-HumanSeg](./contrib/PP-HumanSeg)，以及大规模视频会议数据集[PP-HumanSeg14K](./contrib/PP-HumanSeg/paper.md#pp-humanseg14k-a-large-scale-teleconferencing-video-dataset)。
* [2021-10-11] PaddleSeg 2.3版本发布交互式分割工具EISeg v0.3，开源两种[Matting](./contrib/Matting)算法，以及分割高阶功能[模型蒸馏](./slim/distill)和[模型量化](./slim/quant)方案。


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
* 欢迎加入PaddleSeg 微信群
<div align="center">
<img src="https://user-images.githubusercontent.com/48433081/163670184-43cfb3ae-2047-4ba3-8dae-6c02090dd177.png"  width = "200" />  
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
            <li>ENet</li>
            <li>CCNet</li>
            <li>DDRNet</li>
            <li>GloRe</li>
            <li>PP-LiteSeg :star:</li>
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
            <li>MultiClassFocal</li>
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
            <li>PP-Matting</li>
            <li>DIM</li>
            <li>MODNet</li>
            <li>PP-HumanMatting</li>
        </ul>
        <b>人像分割</b><br>
        <ul>
            <li>PP-HumanSeg</li>
        </ul>
        <b>3D Medical Segmentation</b><br>
        <ul>
          <li>VNet</li>
        </ul>
        <b>Cityscapes SOTA</b><br>
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



## Tutorials <img src="./docs/images/teach.png" width="30"/>

* [Installation Guide](./docs/install.md)
* [Quick Start](./docs/whole_process.md)

*  Data Preparation
   * [Annotated Data Preparation](./docs/data/marker/marker.md)
   * [Annotation Tutorial](./docs/data/transform/transform.md)
   * [Custom Dataset](./docs/data/custom/data_prepare.md)

* [Model Training](/docs/train/train.md)
* [Model Evaluation](./docs/evaluation/evaluate/evaluate.md)
* [Prediction](./docs/predict/predict.md)

* Model Export
    * [Export Inference Model](./docs/model_export.md)
    * [Export ONNX Model](./docs/model_export_onnx.md)

*  Model Deploy
    * [Paddle Inference (Python)](./docs/deployment/inference/python_inference.md)
    * [Paddle Inference (C++)](./docs/deployment/inference/cpp_inference.md)
    * [Paddle Lite](./docs/deployment/lite/lite.md)
    * [Paddle Serving](./docs/deployment/serving/serving.md)
    * [Paddle JS](./docs/deployment/web/web.md)
    * [Benchmark](./docs/deployment/inference/infer_benchmark.md)

*  Model Compression
    * [Quantization](./docs/slim/quant/quant.md)
    * [Distillation](./docs/slim/distill/distill.md)
    * [Prune](./docs/slim/prune/prune.md)

*  Easy API
    * [API Documention](./docs/apis/README.md)
    * [API Tutorial](./docs/api_example.md)
*  Baisc Knowledge
    * [Data Augmentation](./docs/module/data/data.md)
    * [Loss Description](./docs/module/loss/losses_en.md)
*  Advanced Development
    * [Detailed Configuration File](./docs/design/use/use.md)
    * [Create Your Own Model](./docs/design/create/add_new_model.md)
*  Pull Request
    * [PR Tutorial](./docs/pr/pr/pr.md)
    * [PR Style](./docs/pr/pr/style_cn.md)

* [Static Graph Version](./docs/static/static.md)
* [Community](#Community)
* [FAQ](./docs/faq/faq/faq.md)

## Practical Projects
  * [Interactive Segmentation](./EISeg)
  * [Image Matting](./Matting)
  * [PP-HumanSeg](./contrib/PP-HumanSeg)
  * [3D Medical Segmentation](./contrib/MedicalSeg)
  * [Cityscapes SOTA](./contrib/CityscapesSOTA)
  * [Panoptic Segmentation](./contrib/PanopticDeepLab)
  * [CVPR Champion Solution](./contrib/AutoNUE)
  * [Domain Adaptation](./contrib/DomainAdaptation)


## Community <img src="./docs/images/chat.png" width="30"/>

* If you have any problem or suggestion on PaddleSeg, please send us issues through [GitHub Issues](https://github.com/PaddlePaddle/PaddleSeg/issues).
* Welcome to Join PaddleSeg WeChat Group
<div align="center">
<img src="https://user-images.githubusercontent.com/48433081/163670184-43cfb3ae-2047-4ba3-8dae-6c02090dd177.png"  width = "200" />  
</div>

## License

PaddleSeg is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement
* Thanks [jm12138](https://github.com/jm12138) for contributing U<sup>2</sup>-Net.
* Thanks [zjhellofss](https://github.com/zjhellofss) (Fu Shenshen) for contributing Attention U-Net, and Dice Loss.
* Thanks [liuguoyu666](https://github.com/liguoyu666), [geoyee](https://github.com/geoyee) for contributing U-Net++ and U-Net3+.
* Thanks [yazheng0307](https://github.com/yazheng0307) (LIU Zheng) for contributing quick-start document.
* Thanks [CuberrChen](https://github.com/CuberrChen) for contributing STDC(rethink BiSeNet), PointRend and DetailAggregateLoss.
* Thanks [stuartchen1949](https://github.com/stuartchen1949) for contributing SegNet.
* Thanks [justld](https://github.com/justld) (Lang Du) for contributing DDRNet, CCNet, ESPNetV2, DMNet, ENCNet, HRNet_W48_Contrast, FastFCN, BiSeNetV1, SECrossEntropyLoss and PixelContrastCrossEntropyLoss.
* Thanks [Herman-Hu-saber](https://github.com/Herman-Hu-saber) (Hu Huiming) for contributing ESPNetV2.
* Thanks [zhangjin12138](https://github.com/zhangjin12138) for contributing RandomCenterCrop.
* Thanks [simuler](https://github.com/simuler) for contributing ESPNetV1.
* Thanks [ETTR123](https://github.com/ETTR123)(Zhang Kai) for contributing ENet, PFPNNet.

## Citation
If you find our project useful in your research, please consider citing:

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

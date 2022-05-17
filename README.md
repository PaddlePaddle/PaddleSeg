English | [简体中文](README_CN.md)

<div align="center">

<p align="center">
  <img src="./docs/images/paddleseg_logo.png" align="middle" width = "500" />
</p>

**A High-Efficient Development Toolkit for Image Segmentation based on [PaddlePaddle](https://github.com/paddlepaddle/paddle).**

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

</ul>


## Introduction

PaddleSeg is an end-to-end high-efficent development toolkit for image segmentation based on PaddlePaddle, which  helps both developers and researchers in the whole process of designing segmentation models, training models, optimizing performance and inference speed, and deploying models. A lot of well-trained models and various real-world applications in both industry and academia help users conveniently build hands-on experiences in image segmentation.

* #### Four segmentation areas: semantic segmentation, interactive segmentation, panoptic segmentation and image matting.

<div align="center">
<img src="https://user-images.githubusercontent.com/53808988/130562378-64d0c84a-9c3f-4ae4-93f7-bdc0c8e0238e.gif"  width = "2000" />  
</div>


---------------

 * #### Various applications in autonomous driving, medical segmentation, remote sensing, quality inspection, and other scenarios.

<div align="center">
<img src="https://user-images.githubusercontent.com/53808988/130562234-bdf79d76-8566-4e06-a3a9-db7719e63385.gif"  width = "2000" />  
</div>


---------------


## Features

* <img src="./docs/images/f1.png" width="20"/> **High-Performance Model**: Based on the high-performance backbone trained by semi-supervised label knowledge distillation scheme ([SSLD]((https://paddleclas.readthedocs.io/zh_CN/latest/advanced_tutorials/distillation/distillation.html#ssld))), combined with the state of the art segmentation technology, we provide 80+ high-quality pre-training models, which are better than other open-source implementations.

* <img src="./docs/images/f2.png" width="20"/> **Modular Design**: PaddleSeg supports 40+ mainstream *segmentation networks*, developers can start based on actual application scenarios and assemble diversified training configurations combined with modular design of *data enhancement strategies*, *backbone networks*, *loss functions* and other different components to meet different performance and accuracy requirements.

* <img src="./docs/images/f3.png" width="20"/> **High Efficiency**: PaddleSeg provides multi-process asynchronous I/O, multi-card parallel training, evaluation, and other acceleration strategies, combined with the memory optimization function of the PaddlePaddle, which can greatly reduce the training overhead of the segmentation model, all this allowing developers to lower cost and more efficiently train image segmentation model.



## Overview <img src="./docs/images/model.png" width="20"/>

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Models</b>
      </td>
      <td colspan="2">
        <b>Components</b>
      </td>
      <td>
        <b>Projects</b>
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
        <b>Backbones</b><br>
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
        <b>Losses</b><br>
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
        <b>Metrics</b><br>
          <ul>
            <li>mIoU</li>
            <li>Accuracy</li>
            <li>Kappa</li>
            <li>Dice</li>
            <li>AUC_ROC</li>
          </ul>  
      </td>
      <td>
        <b>Datasets</b><br>
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
        <b>Data Augmentation</b><br>
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
        <b>Interactive Segmentation</b><br>
          <ul>
            <li>EISeg</li>
            <li>RITM</li>
            <li>EdgeFlow</li>
           </ul>
       <b>Image Matting</b><br>
        <ul>
            <li>PP-Matting</li>
            <li>DIM</li>
            <li>MODNet</li>
            <li>PP-HumanMatting</li>
        </ul>
        <b>Human Segmentation</b><br>
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
        <b>Panoptic Segmentation</b><br>
          <ul>
            <li>Panoptic-DeepLab</li>
          </ul>
        <b>CVPR Champion</b><br>
        <ul>
            <li>MLA Transformer</li>
        </ul>
        <b>Domain Adaption</b><br>
        <ul>
            <li>PixMatch</li>
        </ul>
      </td>  
    </tr>


</td>
    </tr>
  </tbody>
</table>

## Model Zoo

The relationship between mIoU and FLOPs of representative architectures and backbones. See [Model Zoo Overview](./docs/model_zoo_overview.md) for more details.

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
  * [3D ](./contrib/MedicalSeg)
  * [Cityscapes SOTA](./contrib/CityscapesSOTA)
  * [Panoptic Segmentation](./contrib/PanopticDeepLab)
  * [CVPR Champion Solution](./contrib/AutoNUE)
  * [Domain Adaptation](./contrib/DomainAdaptation)


## Community <img src="./docs/images/chat.png" width="30"/>

* If you have any problem or suggestion on PaddleSeg, please send us issues through [GitHub Issues](https://github.com/PaddlePaddle/PaddleSeg/issues).
* Welcome to Join PaddleSeg WeChat Group
<div align="center">
<img src="https://user-images.githubusercontent.com/48433081/168747513-4bd9e926-d907-40eb-be27-90f9892907b2.png"  width = "200" />  
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
    author={PaddlePaddle Contributors},
    howpublished = {\url{https://github.com/PaddlePaddle/PaddleSeg}},
    year={2019}
}
```

English | [简体中文](README_CN.md)

# PaddleSeg

[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleSeg.svg)](https://github.com/PaddlePaddle/PaddleSeg/releases)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)
## PaddleSeg has released the new version including the following features:

* We published a paper on portrait segmentation named [PP-HumanSeg](./contrib/PP-HumanSeg/paper.md), and release Semantic Connectivity-aware Learning (SCL) framework and a Large-Scale Teleconferencing Video Dataset.
* We published a paper on interactive segmentation named [EdgeFlow](https://arxiv.org/abs/2109.09406), in which the proposed approach achieved SOTA performance on several well-known datasets, and upgraded the interactive annotation tool, [EISeg](./EISeg).
* We released two [Matting](./contrib/Matting) algorithms, DIM and MODNet, which achieve extremely fine-grained segmentation.
* We provided advanced features on segmentation model compression, [Knowlede Distillation](./slim/distill) and [Model Quantization](./slim/quant), which accelerate model inference on multi-devices deployment.

## PaddleSeg Introduction

Welcome to PaddleSeg! PaddleSeg is an end-to-end image segmentation development kit developed based on [PaddlePaddle](https://www.paddlepaddle.org.cn), which covers a large number of high-quality segmentation models in different directions such as *high-performance* and *lightweight*. With the help of modular design, we provide two application methods: *Configuration Drive* and *API Calling*. So one can conveniently complete the entire image segmentation application from training to deployment through configuration calls or API calls.

* ### PaddleSeg provides four image segmentation capabilities: semantic segmentation, interactive segmentation, panoptic segmentation and Matting.

<div align="center">
<img src="https://user-images.githubusercontent.com/53808988/130562378-64d0c84a-9c3f-4ae4-93f7-bdc0c8e0238e.gif"  width = "2000" />  
</div>


---------------

 * ### PaddleSeg is widely used in autonomous driving, medical, quality inspection, inspection, entertainment, and other scenarios.

<div align="center">
<img src="https://user-images.githubusercontent.com/53808988/130562234-bdf79d76-8566-4e06-a3a9-db7719e63385.gif"  width = "2000" />  
</div>


---------------


## Core Features

* <img src="./docs/images/f1.png" width="20"/> **High-Performance Model**: Based on the high-performance backbone trained by Baidu's self-developed [semi-supervised label knowledge distillation scheme (SSLD)](https://paddleclas.readthedocs.io/zh_CN/latest/advanced_tutorials/distillation/distillation.html#ssld), combined with the state of the art segmentation technology, we provide 50+ high-quality pre-training models, which are better than other open-source implementations.

* <img src="./docs/images/f2.png" width="20"/> **Modular Design**: PaddleSeg support 15+ mainstream *segmentation networks*, developers can start based on actual application scenarios and assemble diversified training configurations combined with modular design of *data enhancement strategies*, *backbone networks*, *loss functions* and other different components to meet different performance and accuracy requirements.

* <img src="./docs/images/f3.png" width="20"/> **High Efficiency**: PaddleSeg provides multi-process asynchronous I/O, multi-card parallel training, evaluation, and other acceleration strategies, combined with the memory optimization function of the PaddlePaddle, which can greatly reduce the training overhead of the segmentation model, all this allowing developers to lower cost and more efficiently train image segmentation model.

#### Overview of Kit Structures

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Semantic Segmentation</b>
      </td>
      <td>
        <b>Components</b>
      </td>
      <td>
        <b>Data Augmentation</b>
      </td>
      <td>
        <b>Applications</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
          <li><b>Semantic Segmentation</b></li>
          <ul>
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
            <li>DecoupledSegNet</li>
            <li>EMANet</li>
            <li>ISANet</li>
            <li>DNLNet</li>
            <li>SFNet</li>
            <li>PP-HumanSeg-Lite</li>
            <li>PortraitNet</li>
            <li>STDC</li>
            <li>GINet</li>
            <li>PointRend</li>
            <li>SegNet</li>
            <li>ESPNetV2</li>
            <li>HRNetW48Contrast</li>
            <li>DMNet</li>
            <li>ESPNetV1</li>
            <li>ENCNet</li>
            <li>PFPNNet</li>
            <li>SETR</li>
            <li>MLATransformer</li>
            <li>SegFormer</li>
            <li>SegMenter</li>
        </ul>
      </ul>
      </td>
      <td>
        <ul><li><b>Backbones</b></li>
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
        </ul>
        <ul><li><b>Losses</b></li>
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
            <li>OHEM Edge Attention</li>
            <li>Relax Boundary</li>
            <li>Semantic Connectivity</li>
          </ul>  
        </ul>  
        <ul><li><b>Datasets</b></li>
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
          </ul>  
        </ul>
      </td>
      <td>
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
        <ul>
           <li><b>Interactive Seg</b></li>
        <ul>
            <li>RITM</li>
            <li>EdgeFlow</li>
        </ul>
        <li><b>Panoptic Seg</b></li>
        <ul>
            <li>Panoptic-DeepLab</li>
        </ul>
        <li><b>Image Matting</b></li>
        <ul>
            <li>DIM</li>
            <li>MODNet</li>
            <li>PP-HumanMatting</li>
        </ul>
        </ul>  
      </td>  
    </tr>


</td>
    </tr>
  </tbody>
</table>

## Technical Communication <img src="./docs/images/chat.png" width="30"/>

* If you find any problems or have a suggestion with PaddleSeg, please send us issues through [GitHub Issues](https://github.com/PaddlePaddle/PaddleSeg/issues).
* Welcome to Join PaddleSeg QQ Group
<div align="center">
<img src="./docs/images/QQ_chat.png"  width = "200" />  
</div>

## Model Zoo Overview  <img src="./docs/images/model.png" width="20"/>

See [Model Zoo Overview](./docs/model_zoo_overview.md) for more infomation.

<div align="center">
<img src=https://user-images.githubusercontent.com/30695251/140323107-02ce9de4-c8f4-4f18-88b2-59bd0055a70b.png   />  
</div>



## Dataset

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

## Tutorials <img src="./docs/images/teach.png" width="30"/>

* [Installation](./docs/install.md)
* [Get Started](./docs/whole_process.md)
*  Prepare Datasets
   * [Preparation of Annotation Data](./docs/data/marker/marker.md)
   * [Annotating Tutorial](./docs/data/transform/transform.md)
   * [Custom Dataset](./docs/data/custom/data_prepare.md)

*  Custom Development
    * [Detailed Configuration File](./docs/design/use/use.md)
    * [Create Your Own Model](./docs/design/create/add_new_model.md)
    * [PR Tutorial](./docs/pr/pr/pr.md)
    * [Model Guideline](./docs/pr/pr/style_cn.md)
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

* Model Compression
    * [Quantization](./docs/slim/quant/quant.md)
    * [Distillation](./docs/slim/distill/distill.md)
    * [Prune](./docs/slim/prune/prune.md)

*  API Tutorial
    * [API Documention](./docs/apis/README.md)
    * [API Application](./docs/api_example.md)
*  Description of Important Modules
    * [Data Augmentation](./docs/module/data/data.md)
    * [Loss Description](./docs/module/loss/losses_en.md)
    * [Tricks](./docs/module/tricks/tricks.md)
* Description of Classical Models
    * [DeeplabV3](./docs/models/deeplabv3.md)
    * [UNet](./docs/models/unet.md)
    * [OCRNet](./docs/models/ocrnet.md)
    * [Fast-SCNN](./docs/models/fascnn.md)
* [Static Graph Version](./docs/static/static.md)
* [FAQ](./docs/faq/faq/faq.md)

## Installation

#### step 1. Install PaddlePaddle

System Requirements:
* PaddlePaddle >= 2.0.0
* Python >= 3.6+

Highly recommend you install the GPU version of PaddlePaddle, due to the large overhead of segmentation models, otherwise, it could be out of memory while running the models. For more detailed installation tutorials, please refer to the official website of [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/)。


#### step 2. Install PaddleSeg
Support to construct a customized segmentation framework with the *API Calling* method for flexible development.

```shell
pip install paddleseg
```


#### step 3. Download PaddleSeg Repo
Support to complete the whole process segmentation application with *Configuration Drive* method, simple and fast.

```shell
git clone https://github.com/PaddlePaddle/PaddleSeg
```

#### step 4. Verify Installation
Run the following command. If you can train normally, you have installed it successfully.

```shell
python train.py --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml
```


## Practical Cases

* [PP-HumanSeg](./contrib/PP-HumanSeg)
* [Cityscapes SOTA](./contrib/CityscapesSOTA)
* [Panoptic Segmentation](./contrib/PanopticDeepLab)
* [CVPR Champion Solution](./contrib/AutoNUE)
* [Interactive Segmentation](./EISeg)
* [Matting](./contrib/Matting)
* [Domain Adaptation](./contrib/DomainAdaptation)

## Feedbacks and Contact
* The dynamic version is still under development, if you find any issue or have an idea on new features, please don't hesitate to contact us via [GitHub Issues](https://github.com/PaddlePaddle/PaddleSeg/issues).
* PaddleSeg User Group (QQ): 1004738029 or 850378321 or 793114768

## Acknowledgement
* Thanks [jm12138](https://github.com/jm12138) for contributing U<sup>2</sup>-Net.
* Thanks [zjhellofss](https://github.com/zjhellofss) (Fu Shenshen) for contributing Attention U-Net, and Dice Loss.
* Thanks [liuguoyu666](https://github.com/liguoyu666), [geoyee](https://github.com/geoyee) for contributing U-Net++ and U-Net3+.
* Thanks [yazheng0307](https://github.com/yazheng0307) (LIU Zheng) for contributing quick-start document.
* Thanks [CuberrChen](https://github.com/CuberrChen) for contributing STDC(rethink BiSeNet), PointRend and DetailAggregateLoss.
* Thanks [stuartchen1949](https://github.com/stuartchen1949) for contributing SegNet.
* Thanks [justld](https://github.com/justld) (Lang Du) for contributing ESPNetV2, DMNet, ENCNet, HRNet_W48_Contrast, SECrossEntropyLoss and PixelContrastCrossEntropyLoss.
* Thanks [Herman-Hu-saber](https://github.com/Herman-Hu-saber) (Hu Huiming) for contributing ESPNet.
* Thanks [zhangjin12138](https://github.com/zhangjin12138) for contributing RandomCenterCrop.
* Thanks [simuler](https://github.com/simuler) for contributing ESPNetV1.
* Thanks [ETTR123](https://github.com/ETTR123)(Zhang Kai) for contributing PFPNNet.

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

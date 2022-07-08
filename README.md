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



## <img src="./docs/images/seg_news_icon.png" width="20"/> News
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


## <img src="https://user-images.githubusercontent.com/48054808/157795569-9fc77c85-732f-4870-9be0-99a7fe2cff27.png" width="20"/> Introduction

PaddleSeg is an end-to-end high-efficent development toolkit for image segmentation based on PaddlePaddle, which  helps both developers and researchers in the whole process of designing segmentation models, training models, optimizing performance and inference speed, and deploying models. A lot of well-trained models and various real-world applications in both industry and academia help users conveniently build hands-on experiences in image segmentation.

<div align="center">
<img src="https://github.com/shiyutang/files/raw/2bb2aebaaec36f54953c7e4a96cb84c90336e4c1/ezgif.com-gif-maker%20(3).gif"  width = "800" />  
</div>

<div align="center">
<img src="https://github.com/shiyutang/files/raw/main/ezgif.com-gif-maker%20(3).gif"  width = "800" />  
</div>



## <img src="./docs/images/feature.png" width="20"/> Features

* **High-Performance Model**: Following the state of the art segmentation methods and use the high-performance backbone trained by semi-supervised label knowledge distillation scheme ([SSLD]((https://paddleclas.readthedocs.io/zh_CN/latest/advanced_tutorials/distillation/distillation.html#ssld))), we provide 40+ models and 140+ high-quality pre-training models, which are better than other open-source implementations.

* **High Efficiency**: PaddleSeg provides multi-process asynchronous I/O, multi-card parallel training, evaluation, and other acceleration strategies, combined with the memory optimization function of the PaddlePaddle, which can greatly reduce the training overhead of the segmentation model, all this allowing developers to lower cost and more efficiently train image segmentation model.

* **Modular Design**: We desigin PaddleSeg with the modular design philosophy. Therefore, based on actual application scenarios, developers can assemble diversified training configurations with *data enhancement strategies*, *segmentation models*, *backbone networks*, *loss functions* and other different components to meet different performance and accuracy requirements.

* **Complete Flow**: PaddleSeg support image labeling, model designing, model training, model compression and model deployment. With the help of PaddleSeg, developers can easily finish all taskes.

<div align="center">
<img src="https://user-images.githubusercontent.com/14087480/176402154-390e5815-1a87-41be-9374-9139c632eb66.png" width = "800" />  
</div>

## <img src="./docs/images/chat.png" width="20"/> Community

* If you have any questions, suggestions and feature requests, please create an issues in [GitHub Issues](https://github.com/PaddlePaddle/PaddleSeg/issues).
* Welcome to scan the following QR code and join paddleseg wechat group to communicate with us.

<div align="center">
<img src="https://user-images.githubusercontent.com/48433081/163670184-43cfb3ae-2047-4ba3-8dae-6c02090dd177.png"  width = "200" />  
</div>


## <img src="./docs/images/model.png" width="20"/> Overview

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
        <b>Special Case</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        <details><summary><b>Semantic Segmentation</b></summary>
          <ul>
            <li>PP-LiteSeg </li>
            <li>DeepLabV3P  </li>
            <li>OCRNet  </li>
            <li>MobileSeg  </li>
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
        <details><summary><b>Interactive Segmentation</b></summary>
          <ul>
            <li>EISeg</li>
            <li>RITM</li>
            <li>EdgeFlow</li>
          </ul>
        </details>
        <details><summary><b>Image Matting</b></summary>
          <ul>
              <li>PP-Matting</li>
              <li>DIM</li>
              <li>MODNet</li>
              <li>PP-HumanMatting</li>
          </ul>
        </details>
        <details><summary><b>Panoptic Segmentation</b></summary>
          <ul>
            <li>Panoptic-DeepLab</li>
          </ul>
        </details>
      </td>
      <td>
        <details><summary><b>Backbone</b></summary>
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
        <details><summary><b>Loss</b></summary>
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
        <details><summary><b>Metrics</b></summary>
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
        <details><summary><b>Dataset</b></summary>
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
        <details><summary><b>Data Augmentation</b></summary>
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
        <details><summary><b>Human Segmentation</b></summary>
          <ul>
              <li>PP-HumanSeg</li>
          </ul>
        </details>
        <details><summary><b>3D Medical Segmentation</b></summary>
          <ul>
            <li>VNet</li>
          </ul>
        </details>
        <details><summary><b>Cityscapes SOTA Model</b></summary>
          <ul>
              <li>HMSA</li>
          </ul>
        </details>
        <details><summary><b>CVPR Champion Model</b></summary>
          <ul>
              <li>MLA Transformer</li>
          </ul>
        </details>
        <details><summary><b>Domain Adaptation</b></summary>
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


## <img src="https://user-images.githubusercontent.com/48054808/157801371-9a9a8c65-1690-4123-985a-e0559a7f9494.png" width="20"/> Industrial Segmentation Models

<details>
<summary><b>High Accuracy Semantic Segmentation Models</b></summary>

#### These models have good performance and costly inference time, so they are designed for GPU and Jetson devices.

| Model    | Backbone | Cityscapes mIoU(%)    |  V100 TRT Inference Speed(FPS)  |  Config File |
|:-------- |:--------:|:---------------------:|:-------------------------------:|:------------:|
| FCN            | HRNet_W18        | 78.97                 | 24.43     | [yml](./configs/fcn/)         |
| FCN            | HRNet_W48        | 80.70                 | 10.16     | [yml](./configs/fcn/)         |
| DeepLabV3      | ResNet50_OS8     | 79.90                 | 4.56      | [yml](./configs/deeplabv3/)   |
| DeepLabV3      | ResNet101_OS8    | 80.85                 | 3.2       | [yml](./configs/deeplabv3/)   |
| DeepLabV3      | ResNet50_OS8     | 80.36                 | 6.58      | [yml](./configs/deeplabv3p/)  |
| DeepLabV3      | ResNet101_OS8    | 81.10                 | *3.94*    | [yml](./configs/deeplabv3p/)  |
| :star2: OCRNet :star2:      | HRNet_w18        | 80.67                 | 13.26     | [yml](./configs/ocrnet/)      |
| OCRNet         | HRNet_w48        | 82.15                 | 6.17      | [yml](./configs/ocrnet/)      |
| CCNet          | ResNet101_OS8    | 80.95                 | 3.24      | [yml](./configs/ccnet/)       |

Note that:
* Test the inference speed on Nvidia GPU V100: use PaddleInference Python API, enable TensorRT, the data type is FP32, the dimension of input is 1x3x1024x2048.

</details>


<details>
<summary><b>Lightweight Semantic Segmentation Models</b></summary>

#### The segmentation accuracy and inference speed of these models are medium. They can be deployed on GPU, X86 CPU and ARM CPU.

| Model    | Backbone | Cityscapes mIoU(%)    |  V100 TRT Inference Speed(FPS)  | Snapdragon 855 Inference Speed(FPS) | Config File |
|:-------- |:--------:|:---------------------:|:-------------------------------:|:-----------------:|:--------:|
| :star2: PP-LiteSeg :star2:      | STDC1         | 77.04               | 69.82           | 17.22       | [yml](./configs/pp_liteseg/)  |
| :star2: PP-LiteSeg :star2:      | STDC2         | 79.04               | 54.53           | 11.75       | [yml](./configs/pp_liteseg/)  |
| BiSeNetV1           | -             | 75.19               | 14.67           | 1.53      |[yml](./configs/bisenetv1/)  |
| BiSeNetV2           | -             | 73.19               | 61.83           | 13.67       |[yml](./configs/bisenet/)  |
| STDCSeg             | STDC1         | 74.74               | 62.24           | 14.51       |[yml](./configs/stdcseg/)  |
| STDCSeg             | STDC2         | 77.60               | 51.15           | 10.95       |[yml](./configs/stdcseg/)  |
| DDRNet_23           | -             | 79.85               | 42.64           | 7.68      |[yml](./configs/ddrnet/)  |
| HarDNet             | -             | 79.03               | 30.3            | 5.44      |[yml](./configs/hardnet/)  |
| SFNet               | ResNet18_OS8  |  78.72              | *10.72*         |   -         | [yml](./configs/sfnet/)  |

Note that:
* Test the inference speed on Nvidia GPU V100: use PaddleInference Python API, enable TensorRT, the data type is FP32, the dimension of input is 1x3x1024x2048.
* Test the inference speed on Snapdragon 855: use PaddleLite CPP API, 1 thread, the dimension of input is 1x3x256x256.

</details>


<details>
<summary><b>Super Lightweight Semantic Segmentation Models</b></summary>

#### These super lightweight semantic segmentation models are designed for X86 CPU and ARM CPU.

| Model    | Backbone | Cityscapes mIoU(%)    |  V100 TRT Inference Speed(FPS)  | Snapdragon 855 Inference Speed(FPS) | Config File |
|:-------- |:--------:|:---------------------:|:-------------------------------:|:-----------------------------------:|:-----------:|
| MobileSeg      | MobileNetV2              | 73.94                 | 67.57          | 27.01   | [yml](./configs/mobileseg/)  |
| :star2: MobileSeg :star2:  | MobileNetV3              | 73.47                 | 67.39          | 32.90   | [yml](./configs/mobileseg/)  |
| MobileSeg      | Lite_HRNet_18            | 70.75                 | *10.5*         | 13.05   | [yml](./configs/mobileseg/)  |
| MobileSeg      | ShuffleNetV2_x1_0        | 69.46                 | *37.09*        | 39.61  | [yml](./configs/mobileseg/)  |
| MobileSeg      | GhostNet_x1_0            | 71.88                 | *35.58*        | 38.74  | [yml](./configs/mobileseg/)  |

Note that:
* Test the inference speed on Nvidia GPU V100: use PaddleInference Python API, enable TensorRT, the data type is FP32, the dimension of input is 1x3x1024x2048.
* Test the inference speed on Snapdragon 855: use PaddleLite CPP API, 1 thread, the dimension of input is 1x3x256x256.

</details>


## <img src="./docs/images/teach.png" width="20"/> Tutorials

### Basic Tutorials

* [Installation](./docs/install.md)
* [Quick Start](./docs/quick_start.md)

*  Data Preparation
   * [Annotated Data Preparation](./docs/data/marker/marker.md)
   * [Annotation Tutorial](./docs/data/transform/transform.md)
   * [Custom Dataset](./docs/data/custom/data_prepare.md)
* [Config Preparation](./docs/prepare_cfg.md)

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

### Advanced Tutorials

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

## License

PaddleSeg is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement
* Thanks [jm12138](https://github.com/jm12138) for contributing U<sup>2</sup>-Net.
* Thanks [zjhellofss](https://github.com/zjhellofss) (Fu Shenshen) for contributing Attention U-Net, and Dice Loss.
* Thanks [liuguoyu666](https://github.com/liguoyu666), [geoyee](https://github.com/geoyee) for contributing U-Net++ and U-Net3+.
* Thanks [yazheng0307](https://github.com/yazheng0307) (LIU Zheng) for contributing quick-start document.
* Thanks [CuberrChen](https://github.com/CuberrChen) for contributing STDC(rethink BiSeNet), PointRend and DetailAggregateLoss.
* Thanks [stuartchen1949](https://github.com/stuartchen1949) for contributing SegNet.
* Thanks [justld](https://github.com/justld) (Lang Du) for contributing UPerNet, DDRNet, CCNet, ESPNetV2, DMNet, ENCNet, HRNet_W48_Contrast, FastFCN, BiSeNetV1, SECrossEntropyLoss and PixelContrastCrossEntropyLoss.
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

English | [简体中文](README_CN.md)

# Image Matting

## Contents
* [Introduction](#Introduction)
* [Update Notes](#Update-Notes)
* [Community](#Community)
* [Models](#Models)
* [Tutorials](#Tutorials)
* [Acknowledgement](#Acknowledgement)
* [Citation](#Citation)


## Introduction

Image Matting is the technique of extracting foreground from an image by calculating its color and transparency.
It is widely used in the film industry to replace background, image composition, and visual effects.
Each pixel in the image will have a value that represents its foreground transparency, called Alpha.
The set of all Alpha values in an image is called Alpha Matte.
The part of the image covered by the mask can be extracted to complete foreground separation.


<p align="center">
<img src="https://user-images.githubusercontent.com/30919197/179751613-d26f2261-7bcf-4066-a0a4-4c818e7065f0.gif" width="100%" height="100%">
</p>

## Update Notes
* 2022.11
  * **Release self developed lite matting SOTA model PP-MattingV2**. Compared with MODNet, the inference speed of PP-MattingV2 is increased by 44.6%, and the average error is decreased by 17.91%.
  * Adjust the document structure and improve the model zoo information.
  * [FastDeploy](https://github.com/PaddlePaddle/FastDeploy) support PP-MattingV2, PP-Matting, PP-HumanMatting and MODNet models.
* 2022.07
  * Release PP-Matting code. Add ClosedFormMatting, KNNMatting, FastMatting, LearningBaseMatting and RandomWalksMatting traditional machine learning algorithms.
  Add GCA model.
  * upport to specify metrics for evaluation. Support to specify metrics for evaluation.
* 2022.04
  * **Release self developed high accuracy matting SOTA model PP-Matting**. Add PP-HumanMatting high-resolution human matting model.
  * Add Grad, Conn evaluation metrics. Add foreground evaluation funciton, which use [ML](https://arxiv.org/pdf/2006.14970.pdf) algorithm to evaluate foreground when prediction or background replacement.
  * Add GradientLoss and LaplacianLoss. Add RandomSharpen, RandomSharpen, RandomReJpeg, RSSN data augmentation strategies.

* 2021.11
  * **Matting Project is released**, which Realizes image matting function.
  * Support Matting models: DIM, MODNet. Support model export and python deployment. Support background replacement function. Support human matting deployment in Android.

## Community

* If you have any questions, suggestions and feature requests, please create an issues in [GitHub Issues](https://github.com/PaddlePaddle/PaddleSeg/issues).
* Welcome to scan the following QR code and join paddleseg wechat group to communicate with us.
<div align="center">
<img src="https://paddleseg.bj.bcebos.com/images/seg_qr_code.png"  width = "200" />  
</div>

## Models

For the widely application scenario -- human matting, we have trained and open source the ** high-quality human matting models**.
According the actual application scenario, you can directly deploy or finetune.

The model zoo includes our self developded high accuracy model PP-Matting and lite model PP-MattingV2.
- PP-Matting is a high accuracy matting model developded by PaddleSeg, which realizes high-resolution image matting under semantic guidance by the design of Guidance Flow.
    For high accuracy, this model is recommended. Two pre-trained models are opened source with 512 and 1024 resolution level.

- PP-MattingV2 is a lite matting SOTA model developed by PaddleSeg. It extracts high-level semantc informating by double-pyramid pool and spatial attention,
    and uses multi-level feature fusion mechanism for both semantic and detail prediciton.

| Model | SAD | MSE | Grad | Conn |Params(M) | FLOPs(G) | FPS | Config File | Checkpoint | Inference Model |
| - | - | -| - | - | - | - | -| - | - | - |
| PP-MattingV2-512   |40.59|0.0038|33.86|38.90| 8.95 | 7.51 | 98.89 |[cfg](../configs/ppmattingv2/ppmattingv2-stdc1-human_512.yml)| [model](https://paddleseg.bj.bcebos.com/matting/models/ppmattingv2-stdc1-human_512.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/ppmattingv2-stdc1-human_512.zip) |
| PP-Matting-512     |31.56|0.0022|31.80|30.13| 24.5 | 91.28 | 28.9 |[cfg](../configs/ppmatting/ppmatting-hrnet_w18-human_512.yml)| [model](https://paddleseg.bj.bcebos.com/matting/models/ppmatting-hrnet_w18-human_512.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/ppmatting-hrnet_w18-human_512.zip) |
| PP-Matting-1024    |66.22|0.0088|32.90|64.80| 24.5 | 91.28 | 13.4(1024X1024) |[cfg](../configs/ppmatting/ppmatting-hrnet_w18-human_1024.yml)| [model](https://paddleseg.bj.bcebos.com/matting/models/ppmatting-hrnet_w18-human_1024.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/ppmatting-hrnet_w18-human_1024.zip) |
| PP-HumanMatting    |53.15|0.0054|43.75|52.03| 63.9 | 135.8 (2048X2048)| 32.8(2048X2048)|[cfg](../configs/human_matting/human_matting-resnet34_vd.yml)| [model](https://paddleseg.bj.bcebos.com/matting/models/human_matting-resnet34_vd.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/pp-humanmatting-resnet34_vd.zip) |
| MODNet-MobileNetV2 |50.07|0.0053|35.55|48.37| 6.5 | 15.7 | 68.4 |[cfg](../configs/modnet/modnet-mobilenetv2.yml)| [model](https://paddleseg.bj.bcebos.com/matting/models/modnet-mobilenetv2.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/modnet-mobilenetv2.zip) |
| MODNet-ResNet50_vd |39.01|0.0038|32.29|37.38| 92.2 | 151.6 | 29.0 |[cfg](../configs/modnet/modnet-resnet50_vd.yml)| [model](https://paddleseg.bj.bcebos.com/matting/models/modnet-resnet50_vd.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/modnet-resnet50_vd.zip) |
| MODNet-HRNet_W18   |35.55|0.0035|31.73|34.07| 10.2 | 28.5 | 62.6 |[cfg](../configs/modnet/modnet-hrnet_w18.yml)| [model](https://paddleseg.bj.bcebos.com/matting/models/modnet-hrnet_w18.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/modnet-hrnet_w18.zip) |
| DIM-VGG16          |32.31|0.0233|28.89|31.45| 28.4 | 175.5| 30.4 |[cfg](../configs/dim/dim-vgg16.yml)| [model](https://paddleseg.bj.bcebos.com/matting/models/dim-vgg16.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/dim-vgg16.zip) |

**Note**：
* The dataset for metrics is composed of PPM-100 and human part of AIM-500, with a total of 195 images, which named [PPM-AIM-195](https://paddleseg.bj.bcebos.com/matting/datasets/PPM-AIM-195.zip).
* The model default input size is (512, 512) while calculating FLOPs and FPS and the GPU is Tesla V100 32G. FPS is calculated base on Paddle Inference.
* DIM is a trimap-base matting method, which metrics are calculated in transition area.
    If no trimap image is provided, the area  0<alpha<255 is used as the transition area after dilation erosion with a radius of 25 pixels.

## Tutorials
* [Online experience](docs/online_demo_en.md)
* [Quick start](docs/quick_start_en.md)
* [Full development](docs/full_develop_en.md)
* [Human matting android deployment](deploy/human_matting_android_demo/README.md)
* [Human matting .NET deployment](https://gitee.com/raoyutian/PaddleSegSharp)
* [Dataset preparation](docs/data_prepare_en.md)
* AI Studio tutorials
  * [The Matting tutorial of PaddleSeg](https://aistudio.baidu.com/aistudio/projectdetail/3876411?contributionType=1)
  * [The image matting tutorial of PP-Matting](https://aistudio.baidu.com/aistudio/projectdetail/5002963?contributionType=1)

## Acknowledgement
* Thanks [Qian bin](https://github.com/qianbin1989228) for their contributons.
* Thanks for the algorithm support of [GFM](https://arxiv.org/abs/2010.16188).

## Citation
```
@article{chen2022pp,
  title={PP-Matting: High-Accuracy Natural Image Matting},
  author={Chen, Guowei and Liu, Yi and Wang, Jian and Peng, Juncai and Hao, Yuying and Chu, Lutao and Tang, Shiyu and Wu, Zewu and Chen, Zeyu and Yu, Zhiliang and others},
  journal={arXiv preprint arXiv:2204.09433},
  year={2022}
}
```

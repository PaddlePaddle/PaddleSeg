English | [简体中文](README_CN.md)

# PaddleSeg（Dynamic Graph）

[![Build Status](https://travis-ci.org/PaddlePaddle/PaddleSeg.svg?branch=master)](https://travis-ci.org/PaddlePaddle/PaddleSeg)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleSeg.svg)](https://github.com/PaddlePaddle/PaddleSeg/releases)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)

Welcome to the dynamic version! PaddleSeg is the first development kit which supports PaddlePaddle 2.0. Currently, we provide an experimental version that allows developers to have full-featured experience on dynamic graph. In the near future, the dynamic version will be set as default, and the static one will be moved to "legacy" directory.

The full-detailed documents and tutorials are coming soon. So far there are minimum tutorials that help you to enjoy the strengths of dynamic version.

## Model Zoo

|Model\Backbone|ResNet50|ResNet101|HRNetw18|HRNetw48|
|-|-|-|-|-|
|[ANN](./configs/ann)|✔|✔|||
|[BiSeNetv2](./configs/bisenet)|-|-|-|-|
|[DANet](./configs/danet)|✔|✔|||
|[Deeplabv3](./configs/deeplabv3)|✔|✔|||
|[Deeplabv3P](./configs/deeplabv3p)|✔|✔|||
|[Fast-SCNN](./configs/fastscnn)|-|-|-|-|
|[FCN](./configs/fcn)|||✔|✔|
|[GCNet](./configs/gcnet)|✔|✔|||
|[GSCNN](./configs/gscnn)|✔|✔|||
|[HarDNet](./configs/hardnet)|-|-|-|-|
|[OCRNet](./configs/ocrnet/)|||✔|✔|
|[PSPNet](./configs/pspnet)|✔|✔|||
|[U-Net](./configs/unet)|-|-|-|-|
|[U<sup>2</sup>-Net](./configs/u2net)|-|-|-|-|
|[Att U-Net](./configs/attention_unet)|-|-|-|-|
|[U-Net++](./configs/unet_plusplus)|-|-|-|-|

## Dataset

- [x] Cityscapes
- [x] Pascal VOC
- [x] ADE20K
- [ ] Pascal Context
- [ ] COCO stuff

## Installation

1. Install PaddlePaddle

System Requirements:
* PaddlePaddle >= 2.0.0rc
* Python >= 3.6+

> Note: the above requirements are for the **dynamic** graph version. If you intent to use the static one, please refers to [here](../README.md).

Highly recommend you install the GPU version of PaddlePaddle, due to large overhead of segmentation models, otherwise it could be out of memory while running the models.

For more detailed installation tutorials, please refer to the official website of [PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-beta/install/index_cn.html)。


### Download PaddleSeg

```
git clone https://github.com/PaddlePaddle/PaddleSeg
```

### Install Dependencies
Install the python dependencies via the following commands，and please make sure execute it at least once in your branch.
```shell
cd PaddleSeg
export PYTHONPATH=`pwd`
# Run the following one on Windows
# set PYTHONPATH=%cd%
pip install -r requirements.txt
```

## Quick Training
```shell
python train.py --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml
```

## Tutorials

* [Get Started](./docs/quick_start.md)
* [Data Preparation](./docs/data_prepare.md)
* [Training Configuration](./configs/)
* [Add New Components](./docs/add_new_model.md)


## Feedbacks and Contact
* The dynamic version is still under development, if you find any issue or have an idea on new features, please don't hesitate to contact us via [GitHub Issues](https://github.com/PaddlePaddle/PaddleSeg/issues).
* PaddleSeg User Group (QQ): 850378321 or 793114768

## Acknowledgement
* Thanks [jm12138](https://github.com/jm12138) for contributing U<sup>2</sup>-Net.
* Thanks [zjhellofss](https://github.com/zjhellofss) (Fu Shenshen) for contributing Attention U-Net, and Dice Loss.

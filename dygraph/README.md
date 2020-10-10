# PaddleSeg（动态图版本）

## 模型库

* ANN
* BiSeNetv2
* DANet
* Deeplabv3
* Deeplabv3p
* Fast-SCNN
* FCN
* GCNet
* [OCRNet](https://github.com/nepeplwu/PaddleSeg/blob/develop/dygraph/configs/ocrnet/README.md)
* PSPNet
* UNet

## 安装

1. 安装PaddlePaddle

版本要求

* PaddlePaddle >= 2.0.0b

* Python >= 3.6+

由于图像分割模型计算开销大，推荐在GPU版本的PaddlePaddle下使用PaddleSeg.

```shell
pip install -U paddlepaddle-gpu
```

2. 下载PaddleSeg代码
```shell
git clone https://github.com/PaddlePaddle/PaddleSeg
```

3. 安装PaddleSeg依赖
通过以下命令安装python包依赖，请确保在该分支上至少执行过一次以下命令：

```
cd PaddleSeg/dygpraph
pip install -r requirements.txt
```

## 使用教程

* [快速入门](./docs/quick_start.md)
* [数据集准备](./docs/data_prepare.md)
* [配置项](./docs/config.md)
* [多尺度预测 & 滑动窗口预测](./docs/infer.md)

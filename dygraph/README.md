# PaddleSeg（动态图版本）

本目录提供了PaddleSeg的动态图版本，目前已经完成了模型训练、评估、数据处理等功能，在未来的版本中，PaddleSeg将会启动默认的动态图模式。目前该目录处于实验阶段，如果您在使用过程中遇到任何问题，请通过issue反馈给我们，我们将会在第一时间跟进处理。

## 模型库

|模型\骨干网络|ResNet50|ResNet101|HRNetw18|HRNetw48|
|-|-|-|-|-|
|[ANN](./configs/ann/README.md)|✔|✔|||
|BiSeNetv2|-|-|-|-|
|DANet|✔|✔|||
|Deeplabv3|✔|✔|||
|Deeplabv3p|✔|✔|||
|[Fast-SCNN](./configs/fastscnn/README.md)|-|-|-|-|
|FCN|||✔|✔|
|GCNet|✔|✔|||
|[OCRNet](./configs/ocrnet/)|||✔|✔|
|PSPNet|✔|✔|||
|UNet|-|-|-|-|

## 数据集

- [x] CityScapes
- [x] Pascal VOC
- [x] ADE20K
- [ ] Pascal Context
- [ ] COCO stuff

## 安装

1. 安装PaddlePaddle

版本要求

* PaddlePaddle >= 2.0.0b

* Python >= 3.6+

由于图像分割模型计算开销大，推荐在GPU版本的PaddlePaddle下使用PaddleSeg.

安装教程请见[PaddlePaddle官网](https://www.paddlepaddle.org.cn/install/quick)。


2. 下载PaddleSeg代码
```shell
git clone https://github.com/PaddlePaddle/PaddleSeg
```

3. 安装PaddleSeg依赖
通过以下命令安装python包依赖，请确保在该分支上至少执行过一次以下命令：

```
cd PaddleSeg/dygpraph
export PYTHONPATH=`pwd`
pip install -r requirements.txt
```

## 训练
```
python3 train.py --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml
```

## 使用教程

* [快速入门](./docs/quick_start.md)
* [数据集准备](./docs/data_prepare.md)
* [配置项](./configs/)

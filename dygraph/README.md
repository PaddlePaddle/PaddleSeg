# PaddleSeg（动态图版本）

本目录提供了PaddleSeg的动态图版本，目前已经完成了模型训练、评估、数据处理等功能，在未来的版本中，PaddleSeg将会启动默认的动态图模式。目前该目录处于实验阶段，如果您在使用过程中遇到任何问题，请通过issue反馈给我们，我们将会在第一时间跟进处理。

## 模型库

|模型\骨干网络|ResNet50|ResNet101|HRNetw18|HRNetw48|
|-|-|-|-|-|
|ANN|✔|✔|||
|BiSeNetv2|-|-|-|-|
|DANet|✔|✔|||
|Deeplabv3|✔|✔|||
|Deeplabv3p|✔|✔|||
|Fast-SCNN|-|-|-|-|
|FCN|||✔|✔|
|GCNet|✔|✔|||
|[OCRNet](./configs/ocrnet/)|||✔|✔|
|PSPNet|✔|✔|||
|UNet|-|-|-|-|

## 数据集

* CityScapes
* Pascal VOC
* ADE20K

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

## 训练
```
python3 train.py --model_name unet \
--dataset OpticDiscSeg \
--input_size 192 192 \
--iters 10 \
--save_interval 1 \
--do_eval \
--save_dir output
```

## 使用教程

* [快速入门](./docs/quick_start.md)
* [数据集准备](./docs/data_prepare.md)
* [配置项](./configs/)
* [APIs](./docs/apis/)
* [多尺度预测 & 滑动窗口预测](./docs/infer.md)

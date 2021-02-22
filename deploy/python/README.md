# PaddleSeg Python 预测部署方案

## 1. 说明

本方案旨在提供一个PaddlePaddle跨平台图像分割模型的Python预测部署方案作为参考，用户通过一定的配置，加上少量的代码，即可把模型集成到自己的服务中，完成图像分割的任务。

## 2. 前置准备

请使用[模型导出工具](../../docs/model_export.md)导出您的模型, 或点击下载我们的[样例模型](https://paddleseg.bj.bcebos.com/dygraph/demo/bisenet_demo_model.tar.gz)用于测试。

接着准备一张测试图片用于试验效果，我们提供了cityscapes验证集中的一张[图片](https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png)用于演示效果，如果您的模型是使用其他数据集训练的，请自行准备测试图片。

## 3. 预测

在终端输入以下命令进行预测:
```shell
python deploy/python/infer.py --config /path/to/deploy.yaml --image_path
```

参数说明如下:
|参数名|用途|是否必选项|默认值|
|-|-|-|-|
|config|配置文件|是|-|
|image_path|预测图片的路径或者目录|是|-|
|batch_size|单卡batch size|否|配置文件中指定值|
|save_dir|保存预测结果的目录|否|output|

*测试样例和预测结果如下*
![cityscape_predict_demo.png](../../docs/images/cityscapes_predict_demo.png)

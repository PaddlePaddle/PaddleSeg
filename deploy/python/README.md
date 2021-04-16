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
|config|**导出模型时生成的配置文件**, 而非configs目录下的配置文件|是|-|
|image_path|预测图片的路径或者目录|是|-|
|use_trt|是否开启TensorRT来加速预测|否|否|
|use_int8|启动TensorRT预测时，是否以int8模式运行|否|否|
|batch_size|单卡batch size|否|配置文件中指定值|
|save_dir|保存预测结果的目录|否|output|
|with_argmax|对预测结果进行argmax操作|否|否|

*测试样例和预测结果如下*
![cityscape_predict_demo.png](../../docs/images/cityscapes_predict_demo.png)

*注意：*
*1. 当使用量化模型预测时，需要同时开启TensorRT预测和int8预测才会有加速效果*
*2. 使用TensorRT需要使用支持TRT功能的Paddle库，请参考[附录](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#whl-release)下载对应的PaddlePaddle安装包，或者参考[源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/fromsource.html)自行编译*

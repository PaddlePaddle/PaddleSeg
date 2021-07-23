简体中文 | [English](README.md)
# 本地Inference部署

## 1. 说明

本方案旨在提供一个PaddlePaddle跨平台图像分割模型的Python预测部署方案作为参考，用户通过一定的配置，加上少量的代码，即可把模型集成到自己的服务中，完成图像分割的任务。

## 2. 前置准备

请参考[文档](../../export/export/model_export.md)导出你的模型, 或点击下载我们的[样例模型](https://paddleseg.bj.bcebos.com/dygraph/demo/bisenet_demo_model.tar.gz)用于测试。

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
![cityscape_predict_demo.png](../../../docs/images/cityscapes_predict_demo.png)

*注意：*
*1. 当使用量化模型预测时，需要同时开启TensorRT预测和int8预测才会有加速效果*
*2. 使用TensorRT需要使用支持TRT功能的Paddle库，请参考[附录](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#whl-release)下载对应的PaddlePaddle安装包，或者参考[源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/fromsource.html)自行编译*

## 4. 计算推理速度
将导出后的模型进行部署后，你可能很关心部署后模型的推理速度。PaddleSeg提供了测试模型推理速度的方法。以下将介绍如何在你的机器上进行推理速度测试。
* 首先，准备测试数据集（建议使用 100 张以上的原始图像，不要使用标注图像）
* 在终端执行以下命令：
```shell
python deploy/python/infer_timer.py \
--config output/BisenetV2/deploy.yaml \
--image_path /path/to/data_dir \
```
> 其中 /path/to/data_dir 指存放图像的路径，请根据实际情况自行修改。
> data_dir 支持单张图像，也支持图像文件目录。
> 进行测速时，我们建议你使用包含多张图像的文件目录以得到更加可靠的推理速度平均值。

### 注意
* 1、如果你所指定的 `image_num` 大于 `image_path` 中所存放的图像总数，将以实际图像数目进行测试。
* 2、本方法仅计算推理速度，不涉及获取分割结果。如果想获取推理得到的分割结果，请按 `3` 中所指示的方法运行。
* 3、本方法仅计算推理耗时，不包括预处理与后处理耗时。
* 4、测试结果与`机器配置`、`显存占用情况`、`图像分辨率`等多种因素相关，结果仅供参考。

参数说明如下:
|参数名|用途|是否必选项|默认值|
|-|-|-|-|
|config|**导出模型时生成的配置文件**, 而非configs目录下的配置文件|是|-|
|image_path|预测图片的路径或者目录|是|-|
|image_num|预测图片的数目，用以计算平均值|否|100|
|use_trt|是否开启TensorRT来加速预测|否|否|
|use_int8|启动TensorRT预测时，是否以int8模式运行|否|否|
|batch_size|单卡batch size|否|配置文件中指定值|
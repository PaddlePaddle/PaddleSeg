# Python Inference部署

## 1. 说明

本文档介绍使用Paddle Inference的Python接口在服务器端(Nvidia GPU或者X86 CPU)部署分割模型。

飞桨针对不同场景，提供了多个预测引擎部署模型（如下图），更多详细信息请参考[文档](https://paddleinference.paddlepaddle.org.cn/product_introduction/summary.html)。

![inference_ecosystem](https://user-images.githubusercontent.com/52520497/130720374-26947102-93ec-41e2-8207-38081dcc27aa.png)

## 2. 前置准备

下载[样例模型](https://paddleseg.bj.bcebos.com/dygraph/demo/bisenet_demo_model.tar.gz)用于测试。如果要使用其他模型，大家可以使用[模型导出工具](../../model_export.md)。

```shell
wget https://paddleseg.bj.bcebos.com/dygraph/demo/bisenet_demo_model.tar.gz
tar zxvf bisenet_demo_model.tar.gz
```

下载cityscapes验证集中的一张[图片](https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png)用于演示效果。如果大家的模型是使用其他数据集训练的，请自行准备测试图片。

```
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png
```

## 3. 预测

在PaddleSeg根目录，执行以下命令进行预测:

```shell
python deploy/python/infer.py \
    --config /path/to/deploy.yaml \
    --image_path /path/to/image/path/or/dir
```

参数说明如下:
|参数名|用途|是否必选项|默认值|
|-|-|-|-|
|config|**导出模型时生成的配置文件**, 而非configs目录下的配置文件|是|-|
|image_path|预测图片的路径或者目录或者文件列表|是|-|
|batch_size|单卡batch size|否|1|
|save_dir|保存预测结果的目录|否|output|
|device|预测执行设备，可选项有'cpu','gpu'|否|'gpu'|
|use_trt|是否开启TensorRT来加速预测|否|False|
|precision|启动TensorRT预测时的数值精度，可选项有'fp32','fp16','int8'|否|'fp32'|
|cpu_threads|使用cpu预测的线程数|否|10|
|enable_mkldnn|是否使用MKL-DNN加速cpu预测|否|False|
|benchmark|是否产出日志，包含环境、模型、配置、性能信息|否|False|
|with_argmax|对预测结果进行argmax操作|否|否|

测试样例的预测结果如下。

![cityscape_predict_demo.png](../../images/cityscapes_predict_demo.png)

**注意**

1. 如果使用TensorRT预测，需要安装支持TRT功能的Paddle库。Paddle支持`cuda10.1+cudnn7+trt6.0.1.5`和`cuda10.2+cudnn8.1+trt7.1.3.4`两种版本，大家可以根据实际情况选择，通过如下链接进行下载。
```
https://paddle-inference-dist.bj.bcebos.com/tensorrt_test/cuda10.1-cudnn7.6-trt6.0.tar
https://paddle-inference-dist.bj.bcebos.com/tensorrt_test/cuda10.2-cudnn8.0-trt7.1.tgz
```

* 配置安装cuda和cudnn。
* 下载TRT，设置`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<tensorrt_path>`。
* 参考[附录](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#whl-release)下载带有trt的PaddlePaddle安装包或者参考[源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/fromsource.html)自行编译。
* 安装PaddlePaddle。
* 部署模型。

2. 当使用量化模型在GPU上预测时，需要设置device=gpu、use_trt=True、precision=int8。

3. 要开启`--benchmark`的话需要安装auto_log，请参考[安装方式](https://github.com/LDOUBLEV/AutoLog)。

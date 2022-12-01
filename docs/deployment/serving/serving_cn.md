简体中文 | [English](serving.md)
# Paddle Serving部署

## 概述

PaddleSeg训练出来的模型，大家可以使用[Paddle Serving](https://github.com/PaddlePaddle/Serving)进行服务化部署。

本文以一个示例介绍使用Paddle Serving部署的方法，更多使用教程请参考[文档](https://github.com/PaddlePaddle/Serving/blob/v0.6.0/README_CN.md)。


## 环境准备

使用Paddle Serving部署模型，要求在服务器端和客户端进行如下环境准备。大家具体可以参考[文档](https://github.com/PaddlePaddle/Serving/blob/v0.6.0/README_CN.md#%E5%AE%89%E8%A3%85)进行安装。

在服务器端：
* 安装PaddlePaddle (版本>=2.0)
* 安装paddle-serving-app（版本>=0.6.0）
* 安装paddle-serving-server或者paddle-serving-server-gpu （版本>=0.6.0）

```
pip3 install paddle-serving-app==0.6.0

# CPU
pip3 install paddle-serving-server==0.6.0

# GPU环境需要确认环境再选择
pip3 install paddle-serving-server-gpu==0.6.0.post102 #GPU with CUDA10.2 + TensorRT7
pip3 install paddle-serving-server-gpu==0.6.0.post101 # GPU with CUDA10.1 + TensorRT6
pip3 install paddle-serving-server-gpu==0.6.0.post11 # GPU with CUDA10.1 + TensorRT7
```

在客户端：
* 安装paddle-serving-app（版本>=0.6.0）
* 安装paddle-serving-client（版本>=0.6.0）

```
pip3 install paddle-serving-app==0.6.0
pip3 install paddle-serving-client==0.6.0
```

## 准备模型和数据

下载[样例模型](https://paddleseg.bj.bcebos.com/dygraph/demo/bisenet_demo_model.tar.gz)用于测试。如果要使用其他模型，大家可以使用[模型导出工具](../../model_export.md)。

```shell
wget https://paddleseg.bj.bcebos.com/dygraph/demo/bisenet_demo_model.tar.gz
tar zxvf bisenet_demo_model.tar.gz
```

下载cityscapes验证集中的一张[图片](https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png)用于演示效果。如果大家的模型是使用其他数据集训练的，请自行准备测试图片。

```
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png
```

## 转换模型

Paddle Serving部署之前，我们需要对预测模型进行转换，详细信息请参考[文档](https://github.com/PaddlePaddle/Serving/blob/v0.6.0/doc/SAVE_CN.md)。

在准备好环境的客户端机器上，执行如下脚本对样例模型进行转换。

```shell
python -m paddle_serving_client.convert \
    --dirname ./bisenetv2_demo_model \
    --model_filename model.pdmodel \
    --params_filename model.pdiparams
```

执行完成后，当前目录下的serving_server文件夹保存服务端模型和配置，serving_client文件夹保存客户端模型和配置。

## 服务器端开启服务

大家可以使用paddle_serving_server.serve启动RPC服务，详细信息请参考[文档](https://github.com/PaddlePaddle/Serving/blob/v0.6.0/README_CN.md#rpc%E6%9C%8D%E5%8A%A1)。

在服务器端配置好环境、准备保存服务端模型和配置的serving_server文件后，执行如下命令，启动服务。我们在服务器端使用9292端口，服务器ip使用`hostname -i`查看。

```shell
python -m paddle_serving_server.serve \
    --model serving_server \
    --thread 10 \
    --port 9292 \
    --ir_optim
```

## 客户端请求服务

```
cd PaddleSeg/deploy/serving
```

设置serving_client文件的路径、服务器端ip和端口、测试图片的路径，执行如下命令。

```shell
python test_serving.py \
    --serving_client_path path/to/serving_client \
    --serving_ip_port ip:port \
    --image_path path/to/image\
```

执行完成后，分割的图片保存在当前目录的"result.png"。

![cityscape_predict_demo.png](../../images/cityscapes_predict_demo.png)

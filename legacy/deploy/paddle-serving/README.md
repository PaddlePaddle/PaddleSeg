# 通过PaddleServing部署服务

## 1.简介

PaddleServing是Paddle的在线预测服务框架，可以快速部署训练好的模型用于在线预测。更多信息请参考[PaddleServing 主页](https://github.com/PaddlePaddle/Serving)。本文中将通过unet模型示例，展示预测服务的部署和预测过程。

## 2.安装Paddle Serving

目前PaddleServing的正式版本为0.3.2版本。

服务端安装：

```shell
pip install paddle_serving_server==0.3.2 #CPU
pip install paddle_serving_server_gpu==0.3.2.post9 #GPU with CUDA9.0
pip install paddle_serving_server_gpu==0.3.2.post10 #GPU with CUDA10.0
```

客户端安装：

```shell
pip install paddle_serving_client==0.3.2
```

## 3.导出预测模型


通过训练得到一个满足要求的模型后，如果想要将该模型接入到PaddleServing服务，我们需要通过[`pdseg/export_serving_model.py`](../../pdseg/export_serving_model.py)来导出该模型。

该脚本的使用方法和`train.py/eval.py/vis.py`完全一样。

### FLAGS

|FLAG|用途|默认值|备注|
|-|-|-|-|
|--cfg|配置文件路径|None||

### 使用示例

我们使用[训练/评估/可视化](./usage.md)一节中训练得到的模型进行试用，命令如下

```shell
python pdseg/export_serving_model.py --cfg configs/unet_optic.yaml TEST.TEST_MODEL ./saved_model/unet_optic/final
```

预测模型会导出到`freeze_model`目录，包括`serving_server`和`serving_client`两个子目录。

`freeze_model/serving_server`目录下包含了模型文件和serving server端配置文件，`freeze_model/serving_client`目录下包含了serving client端配置文件。

分别将serving_server和serving_client复制到server和client启动的路径下。

本文中导出的unet模型示例[下载](https://paddle-serving.bj.bcebos.com/paddle_seg_demo/seg_unet_demo.tar.gz)。

解压下载后的压缩包，可以得到seving_server和serving_client两个文件夹，用于以下步骤的测试。

## 4.部署预测服务

```shell
python -m paddle_serving_server.serve --model serving_server/ --port 9494 # CPU
python -m paddle_serving_server_gpu.serve --model serving_server --port 9494 --gpu_ids 0 #GPU
```

## 5.执行预测
```shell
python seg_client.py ../../dataset/optic_disc_seg/JPEGImages/N0060.jpg
```
脚本执行之后，会在输入图片所在的目录下生成处理后的图片
示例中为`../../dataset/optic_disc_seg/JPEGImages/N0060_jpg_mask.png`和`../../dataset/optic_disc_seg/JPEGImages/N0060_jpg_result.png`

如果需要使用其他模型进行预测，需要修改预处理部分和后处理部分。

本文中使用的是paddle_serving_app中内置的处理方法，用户可以参考导出模型时生成的deploy.yaml修改预处理部分。

后处理部分SegPostprocess初始化时接收的参数为分割的类别数。

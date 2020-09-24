# 通过PaddleServing部署服务

## 1.简介

PaddleServing是Paddle的在线预测服务框架，可以快速部署训练好的模型用于在线预测。更多信息请参考[PaddleServing 主页](https://github.com/PaddlePaddle/Serving)。本文中将通过unet模型示例，展示预测服务的部署和预测过程。

## 2.安装Paddle Serving

目前PaddleServing的正式版本为0.3.2版本，本文中的示例需要develop版本的paddle_serving_app，请从[链接](https://github.com/PaddlePaddle/Serving/blob/develop/doc/LATEST_PACKAGES.md#app)中下载并安装。

服务端安装：

```shell
pip install paddle_serving_server==0.3.2 #CPU
pip install paddle_serving_server_gpu==0.3.2.post9 #GPU with CUDA9.0
pip install paddle_serving_server_gpu==0.3.2.post9 #GPU with CUDA10.0
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

## 4.部署预测服务

```shell
python -m paddle_serving_server.serve --model unet_model/ --port 9494 # CPU
python -m paddle_serving_server_gpu.serve --model unet_model --port 9494 --gpu_ids 0 #GPU
```

## 5.执行预测

```python
#seg_client.py
from paddle_serving_client import Client
from paddle_serving_app.reader import Sequential, File2Image, Resize, Transpose, BGR2RGB, SegPostprocess, Normalize, Div
import sys
import cv2

client = Client()
client.load_client_config("unet_client/serving_client_conf.prototxt")
client.connect(["127.0.0.1:9494"])

preprocess = Sequential([
    File2Image(), Resize(
        (512, 512), interpolation=cv2.INTER_LINEAR), Div(255.0),
    Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], False), Transpose((2, 0, 1))
])

postprocess = SegPostprocess(2)

filename = "N0060.jpg"
im = preprocess(filename)
fetch_map = client.predict(feed={"image": im}, fetch=["transpose_1.tmp_0"])
fetch_map["filename"] = filename
postprocess(fetch_map)
```

脚本执行之后，当前目录下生成处理后的图片。

完整的部署示例请参考PaddleServing的[unet示例](https://github.com/PaddlePaddle/Serving/tree/develop/python/examples/unet_for_image_seg)。

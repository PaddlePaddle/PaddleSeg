简体中文 | [English](model_export.md)

# 导出预测模型

PaddleSeg训练好模型后，需要将模型导出为预测模型，才可以进行模型部署。

本教程提供一个示例介绍模型导出的过程。

## 1. 获取预训练莫模型

本示例中，我们使用BiseNetV2模型，大家执行如下命令或者点击[链接](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/bisenet_cityscapes_1024x1024_160k/model.pdparams)下载模型预训练权重。

```shell
mkdir bisenet && cd bisenet
wget https://paddleseg.bj.bcebos.com/dygraph/cityscapes/bisenet_cityscapes_1024x1024_160k/model.pdparams
cd ..
```

## 2. 导出预测模型

确保正确安装PaddleSeg后，在PaddleSeg目录下执行如下命令，则预测模型会保存在output文件夹。

```shell
# 设置1张可用的卡
export CUDA_VISIBLE_DEVICES=0
# windows下请执行以下命令
# set CUDA_VISIBLE_DEVICES=0
python export.py \
       --config configs/bisenet/bisenet_cityscapes_1024x1024_160k.yml \
       --model_path bisenet/model.pdparams \
       --save_dir output
       --input_shape 1 3 512 1024
```

### 导出脚本参数解释

|参数名|用途|是否必选项|默认值|
|-|-|-|-|
|config|配置文件|是|-|
|model_path|预训练权重的路径|否|配置文件中指定的预训练权重路径|
|save_dir|保存预测模型的路径|否|output|
|input_shape| 设置导出模型的输入shape，比如传入`--input_shape 1 3 1024 1024`。如果不设置input_shape，默认导出模型的输入shape是[-1, 3, -1, -1] | 否（ 如果预测shape固定，建议指定input_shape参数 ） | None |
|with_softmax|在网络末端添加softmax算子。由于PaddleSeg组网默认返回logits，如果想要部署模型获取概率值，可以置为True|否|False|
|without_argmax|是否不在网络末端添加argmax算子。由于PaddleSeg组网默认返回logits，为部署模型可以直接获取预测结果，我们默认在网络末端添加argmax算子|否|False|

**如果导出出现和shape相关的问题，请尝试指定input_shape。**

## 3. 预测模型文件

如下是导出的预测模型文件。

```shell
output
  ├── deploy.yaml            # 部署相关的配置文件，主要说明数据预处理的方式
  ├── model.pdmodel          # 预测模型的拓扑结构文件
  ├── model.pdiparams        # 预测模型的权重文件
  └── model.pdiparams.info   # 参数额外信息，一般无需关注
```

导出预测模型后，我们可以使用以下方式部署模型：

|部署场景|使用预测库|教程|
|-|-|-|
|服务器端(Nvidia GPU和X86 CPU) Python部署|Paddle Inference|[文档](../deploy/python/)|
|服务器端(Nvidia GPU和X86 CPU) C++端部署|Paddle Inference|[文档](../deploy/cpp/)|
|移动端部署|Paddle Lite|[文档](../deploy/lite/)|
|服务化部署|Paddle Serving|[文档](../deploy/serving/)|
|前端部署|Paddle JS|[文档](../deploy/web/)|

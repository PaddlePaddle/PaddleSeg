简体中文 | [English](model_export.md)

# 导出预测模型

使用PaddleSeg训练好模型后，我们将模型导出为预测模型、使用预测库进行部署，可以实现更快的推理速度。

本教程基于一个示例介绍模型导出的过程。

## 1. 获取模型权重

完成模型训练后，精度最高的模型权重保存在`path/to/save/best_model/model.pdparams`。

本示例为了演示方便，大家执行如下命令或者点击[链接](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k/model.pdparams)下载PP-LiteSeg模型训练好的权重。

```shell
wget https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k/model.pdparams
```

## 2. 导出预测模型

在PaddleSeg根目录下，执行如下命令，导出预测模型，保存在`output/inference_model`目录。

```shell
python tools/export.py \
       --config configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k.yml \
       --model_path model.pdparams \
       --save_dir output/inference_model
```

**导出脚本参数解释：**

|参数名|用途|是否必选项|默认值|
|-|-|-|-|
|config        | 配置文件的路径       | 是 | - |
|model_path    | 模型权重的路径      | 否 | - |
|save_dir      | 预测模型保存的目录  | 否 | `./output/inference_model` |
|input_shape   | 设置模型的输入shape (`N*C*H*W`)，比如传入`--input_shape 1 3 1024 1024`。如果不设置input_shape，默认导出模型的输入shape是`[-1, 3, -1, -1]`。 预测shape固定时，建议指定input_shape参数。 | 否 | None |
|output_op     | 设置在模型末端添加的输出算子，支持[`argmax`, `softmax`, `none`]。PaddleSeg模型默认返回logits (`N*C*H*W`)；添加`argmax`算子，可以得到每个像素的分割类别，结果的维度是`N*H*W`、数据类型是`int32`；添加`softmax`算子，可以得到每个像素每类的概率，结果的维度是`N*C*H*W`、数据类型是`float32` | 否 | argmax |
|with_softmax  | 即将废弃的输入参数，建议使用`--output_op`。在网络末端添加softmax算子。由于PaddleSeg组网默认返回logits，如果想要部署模型获取概率值，可以置为True | 否 | False |
|without_argmax| 即将废弃的输入参数，建议使用`--output_op`。由于PaddleSeg组网默认返回logits，为部署模型可以直接获取预测结果，我们默认在网络末端添加argmax算子。如果设置`--without_argmax`，则不会在网络末端添加argmax算子。 | 否 | False |

注意：
* 如果部署模型时，出现和shape相关的问题，请尝试指定input_shape。

## 3. 预测模型文件

如下是导出的预测模型文件。

```shell
output/inference_model
  ├── deploy.yaml            # 部署相关的配置文件，主要说明数据预处理方式等信息
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

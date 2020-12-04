# PaddleSeg量化模型部署方案

## 1. 说明

本方案旨在提供一个PaddeSeg量化模型使用TensorRT的`Python`预测部署方案作为参考，用户通过一定的配置，加上少量的代码，即可把模型集成到自己的服务中，完成图像分割的任务。

## 2. 环境准备

* 参考[编译安装文档](../../../deploy/python/docs/compile_paddle_with_tensorrt.md)，编译支持TensorRT的Paddle安装包并安装。

## 3. 开始预测

### 3.1 准备预测模型

请参考[模型量化](../)训练并导出相应的量化模型

模型导出的目录通常包括三个文件:

```
├── model #  模型文件
├── params # 参数文件
└── deploy.yaml # 配置文件，用于C++或Python预测
```

配置文件的主要字段及其含义如下:
```yaml
DEPLOY:
    # 是否使用GPU预测
    USE_GPU: 1
    # 模型和参数文件所在目录路径
    MODEL_PATH: "freeze_model"
    # 模型文件名
    MODEL_FILENAME: "model"
    # 参数文件名
    PARAMS_FILENAME: "params"
    # 预测图片的的标准输入尺寸，输入尺寸不一致会做resize
    EVAL_CROP_SIZE: (2049, 1025)
    # 均值
    MEAN: [0.5, 0.5, 0.5]
    # 方差
    STD: [0.5, 0.5, 0.5]
    # 分类类型数
    NUM_CLASSES: 19
    # 图片通道数
    CHANNELS : 3
    # 预测模式，支持 NATIVE 和 ANALYSIS
    PREDICTOR_MODE: "ANALYSIS"
    # 每次预测的 batch_size
    BATCH_SIZE : 3
```

### 3.2 执行预测程序

```bash
python infer.py --conf=/path/to/deploy.yaml --input_dir=/path/to/images_directory
```
参数说明如下:

| 参数 | 是否必须|含义 |
|-------|-------|----------|
| conf | Yes|模型配置的Yaml文件路径 |
| input_dir |Yes| 需要预测的图片目录 |
| save_dir | No|预测结果的保存路径，默认为output|
| ext | No| 所支持的图片格式，有多种格式时以'\|'分隔，默认为'.jpg\|.jpeg'|
| use_int8 |No| 是否是否Int8预测 |

运行后程序会扫描`input_dir` 目录下所有指定格式图片，并生成`可视化的结果`。

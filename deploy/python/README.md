# PaddleSeg Python 预测部署方案

本文档旨在提供一个`PaddlePaddle`跨平台图像分割模型的`Python`预测部署方案，用户通过一定的配置，加上少量的代码，即可把模型集成到自己的服务中，完成图像分割的任务。

## 前置条件

* Python2.7+，Python3+
* pip，pip3

## 主要目录和文件

```
├── infer.py #  核心代码，完成分割模型的预测以及结果可视化
├── requirements.txt # 依赖的Python包
└── README.md # 说明文档
```

### Step1:安装PaddlePaddle

如何选择合适版本的`PaddlePaddle`版本进行安装，可参考: [PaddlePaddle安装教程](https://www.paddlepaddle.org.cn/install/doc/)

### Step2:安装Python依赖包

2.1 在**当前**目录下, 使用`pip`安装`Python`依赖包
```bash
pip install -r requirements.txt
```

2.2 安装`OpenCV` 相关依赖库
预测代码中需要使用`OpenCV`，所以还需要`OpenCV`安装相关的动态链接库。
以`Ubuntu` 和`CentOS` 为例，命令如下:

`Ubuntu`下安装相关链接库：
```bash
apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
```

CentOS 下安装相关链接库：
```bash
yum install -y libXext libSM libXrender
```
### Step3:预测
进行预测前, 请使用[模型导出工具](../../docs/model_export.md) 导出您的模型(或点击下载我们的[人像分割样例模型](https://bj.bcebos.com/paddleseg/inference/human_freeze_model.zip)用于测试)。

导出的模型目录通常包括三个文件，除了模型文件`models` 和参数文件`params`，还会生成对应的配置文件`deploy.yaml`用于`C++`和`Python` 预测, 主要字段及其含义如下:
```yaml
DEPLOY:
    # 是否使用GPU预测
    USE_GPU: 1
    # 模型和参数文件所在目录路径
    MODEL_PATH: "/root/projects/models/deeplabv3p_xception65_humanseg"
    # 模型文件名
    MODEL_FILENAME: "__model__"
    # 参数文件名
    PARAMS_FILENAME: "__params__"
    # 预测图片的的标准输入尺寸，输入尺寸不一致会做resize
    EVAL_CROP_SIZE: (513, 513)
    # 均值
    MEAN: [0.5, 0.5, 0.5]
    # 方差
    STD: [0.5, 0.5, 0.5]
    # 分类类型数
    NUM_CLASSES: 2
    # 图片通道数
    CHANNELS : 3
    # 预测模式，支持 NATIVE 和 ANALYSIS
    PREDICTOR_MODE: "ANALYSIS"
    # 每次预测的 batch_size
    BATCH_SIZE : 3
```

模型文件就绪后，在终端输入以下命令进行预测:
```
python infer.py --conf=/path/to/deploy.yaml --input_dir/path/to/images_directory --use_pr=False
```
参数说明如下:

| 参数 | 是否必须|含义 |
|-------|-------|----------|
| conf | YES|模型配置的Yaml文件路径 |
| input_dir |YES| 需要预测的图片目录 |
| use_pr |NO|是否使用优化模型，默认为False|

* 优化模型：使用`PaddleSeg 0.3.0`版导出的为优化模型, 此前版本导出的模型即为未优化版本。优化模型把图像的预处理以及后处理部分融入到模型网络中使用`GPU` 完成，相比原来`CPU` 中的处理提升了计算性能。


运行后会扫描`input_dir` 目录下所有指定格式图片，生成`预测mask`和`可视化的结果`。
对于图片`a.jpeg`, `预测mask` 存在`a_jpeg.png` 中，而可视化结果则在`a_jpeg_result.png` 中。

输入样例:
![avatar](../cpp/images/humanseg/demo2.jpeg)

输出结果:  
![avatar](../cpp/images/humanseg/demo2.jpeg_result.png)

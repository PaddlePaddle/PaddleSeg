[English](README.md) | 简体中文
# PaddleSeg C++部署示例

本目录下提供`infer.cc`快速完成PP-LiteSeg在SOPHGO BM1684x板子上加速部署的示例。

## 算能硬件编译FastDeploy环境准备
在部署前，需自行编译基于算能硬件的预测库，参考文档[算能硬件部署环境](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#算能硬件部署环境)

## 生成基本目录文件

该例程由以下几个部分组成
```text
.
├── CMakeLists.txt
├── fastdeploy-sophgo  # 编译文件夹
├── image  # 存放图片的文件夹
├── infer.cc
└── model  # 存放模型文件的文件夹
```

## 编译

### 编译FastDeploy

请参考[SOPHGO部署库编译](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/sophgo.md)编译SDK，编译完成后，将在build目录下生成fastdeploy-sophgo目录。拷贝fastdeploy-sophgo至当前目录

### 拷贝模型文件，以及配置文件至model文件夹
将Paddle模型转换为SOPHGO bmodel模型，转换步骤参考[文档](../README_CN.md#将paddleseg推理模型转换为bmodel模型步骤)

将转换后的SOPHGO bmodel模型文件拷贝至model中

### 准备测试图片至image文件夹
```bash
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png
cp cityscapes_demo.png ./images
```

### 编译example

```bash
cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-sophgo
make
```

## 运行例程

```bash
./infer_demo model images/cityscapes_demo.png
```

## 快速链接
- [PaddleSeg C++ API文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/cpp/html/namespacefastdeploy_1_1vision_1_1segmentation.html)
- [FastDeploy部署PaddleSeg模型概览](../../)
- [Python部署](../python)
- [模型转换](../README_CN.md#将paddleseg推理模型转换为bmodel模型步骤)

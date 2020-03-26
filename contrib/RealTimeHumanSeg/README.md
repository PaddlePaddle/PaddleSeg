# 视频实时图像分割模型C++预测部署

本文档主要介绍实时图像分割模型如何在`Windows`和`Linux`上完成基于`C++`的预测部署。

## C++预测部署编译

### 1. 下载模型
点击右边下载：[模型下载地址](https://paddleseg.bj.bcebos.com/deploy/models/humanseg_paddleseg_int8.zip)

模型文件路径将做为预测时的输入参数，请解压到合适的目录位置。

### 2. 编译
本项目支持在Windows和Linux上编译并部署C++项目，不同平台的编译请参考：
- [Linux 编译](./docs/linux_build.md)
- [Windows 使用 Visual Studio 2019编译](./docs/windows_build.md)

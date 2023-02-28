[English](README.md) | 简体中文
# PaddleSeg CPU-GPU C++部署示例

本目录下提供`infer.cc`快速完成PP-LiteSeg在CPU/GPU，以及GPU上通过Paddle-TensorRT加速部署的示例。

## 1. 说明  
PaddleSeg支持利用FastDeploy在NVIDIA GPU、X86 CPU、飞腾CPU、ARM CPU、Intel GPU(独立显卡/集成显卡)硬件上快速部署Segmentation模型。

## 2. 部署环境准备  
在部署前，需确认软硬件环境，同时下载预编译部署库，参考[FastDeploy安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#FastDeploy预编译库安装)安装FastDeploy预编译库。

## 3. 部署模型准备
在部署前，请准备好您所需要运行的推理模型，你可以选择使用[预导出的推理模型](../README.md)或者[自行导出PaddleSeg部署模型](../README.md)，如果你部署的为**PP-Matting**、**PP-HumanMatting**以及**ModNet**请参考[Matting模型部署](../../../matting)。

## 4. 运行部署示例
以Linux上推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本1.0.0以上(x.x.x>=1.0.0)

```bash  
# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz

# 下载部署示例代码
git clone https://github.com/PaddlePaddle/PaddleSeg.git 
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到develop分支
# git checkout develop
cd PaddleSeg/deploy/fastdeploy/semantic_segmentation/cpp-gpu/cpp

# 编译部署示例
mkdir build && cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# 下载PP-LiteSeg模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz
tar -xvf PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

# 运行部署示例
# CPU推理
./infer_demo PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer cityscapes_demo.png 0
# GPU推理
./infer_demo PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer cityscapes_demo.png 1
# GPU上Paddle-TensorRT推理
./infer_demo PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer cityscapes_demo.png 2
```

运行完成可视化结果如下图所示
<div  align="center">  
<img src="https://user-images.githubusercontent.com/16222477/191712880-91ae128d-247a-43e0-b1e3-cafae78431e0.jpg", width=512px, height=256px />
</div>

- 注意，以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考文档: [如何在Windows中使用FastDeploy C++ SDK](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/use_sdk_on_windows.md)  
- 关于如何通过FastDeploy使用更多不同的推理后端，以及如何使用不同的硬件，请参考文档：[如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)

## 6. 更多指南
- [PaddleSeg C++ API文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/cpp/html/namespacefastdeploy_1_1vision_1_1segmentation.html)
- [FastDeploy部署PaddleSeg模型概览](../../)
- [Python部署](../python)

## 7. 常见问题
- [如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)
- [Intel GPU(独立显卡/集成显卡)的使用](https://github.com/PaddlePaddle/FastDeploy/blob/develop/tutorials/intel_gpu/README.md)
- [编译CPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/cpu.md)
- [编译GPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/gpu.md)
- [编译Jetson部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/jetson.md)

[English](README.md) | 简体中文
# PP-Matting CPU-GPU C++部署示例

本目录下提供`infer.cc`快速完成PP-Matting在CPU/GPU、昆仑芯、华为昇腾以及GPU上通过Paddle-TensorRT加速部署的示例。

## 1. 说明  
PaddleSeg支持利用FastDeploy在NVIDIA GPU、X86 CPU、飞腾CPU、ARM CPU、Intel GPU(独立显卡/集成显卡)硬件上快速部署Matting模型

## 2. 部署环境准备  
在部署前，需确认软硬件环境，同时下载预编译部署库，参考文档[FastDeploy预编译库安装](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install)，**注意** 只有CPU、GPU提供预编译库，华为昇腾以及昆仑芯需要参考以上文档自行编译部署环境。

## 3. 部署模型准备
在部署前，请准备好您所需要运行的推理模型，你可以选择使用[预导出的推理模型](../README.md)或者[自行导出PaddleSeg部署模型](../README.md)。

## 4. 运行部署示例
以Linux上推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本1.0.0以上(x.x.x>=1.0.0)

```bash
# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz

# 下载部署示例代码
git clone https://github.com/PaddlePaddle/PaddleSeg.git 
cd PaddleSeg/deploy/fastdeploy/matting/cpp-gpu/cpp

# 编译部署示例 
mkdir build && cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# 下载PP-Matting模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP-Matting-512.tgz
tar -xvf PP-Matting-512.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/matting_input.jpg
wget https://bj.bcebos.com/paddlehub/fastdeploy/matting_bgr.jpg

# CPU推理
./infer_demo PP-Matting-512 matting_input.jpg matting_bgr.jpg 0
# GPU推理
./infer_demo PP-Matting-512 matting_input.jpg matting_bgr.jpg 1
# GPU上TensorRT推理
./infer_demo PP-Matting-512 matting_input.jpg matting_bgr.jpg 2
# 昆仑芯XPU推理
./infer_demo PP-Matting-512 matting_input.jpg matting_bgr.jpg 3
```
**注意** 以上示例未提供华为昇腾的示例，在编译好昇腾部署环境后，只需改造一行代码，将示例文件中KunlunXinInfer方法的`option.UseKunlunXin()`为`option.UseAscend()`就可以完成在华为昇腾上的推理部署

运行完成可视化结果如下图所示
<div width="840">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852040-759da522-fca4-4786-9205-88c622cd4a39.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852587-48895efc-d24a-43c9-aeec-d7b0362ab2b9.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852116-cf91445b-3a67-45d9-a675-c69fe77c383a.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852554-6960659f-4fd7-4506-b33b-54e1a9dd89bf.jpg">
</div>

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/use_sdk_on_windows.md)

## 5. 更多指南
- [PaddleSeg C++ API文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/cpp/html/namespacefastdeploy_1_1vision_1_1segmentation.html)
- [FastDeploy部署PaddleSeg模型概览](../../)
- [Python部署](../python)

## 6. 常见问题
- [如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)
- [Intel GPU(独立显卡/集成显卡)的使用](https://github.com/PaddlePaddle/FastDeploy/blob/develop/tutorials/intel_gpu/README.md)
- [编译CPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/cpu.md)
- [编译GPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/gpu.md)
- [编译Jetson部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/jetson.md)

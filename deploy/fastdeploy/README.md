# PaddleSeg高性能全场景模型部署方案—FastDeploy

## 目录  
- [FastDeploy介绍](#FastDeploy介绍)  
- [语义分割模型部署](#语义分割模型部署)  
- [Matting模型部署](#Matting模型部署)  
- [常见问题](#常见问题)  

## 1. FastDeploy介绍
<div id="FastDeploy介绍"></div>  

**[⚡️FastDeploy](https://github.com/PaddlePaddle/FastDeploy)**是一款**全场景**、**易用灵活**、**极致高效**的AI推理部署工具，支持**云边端**部署。使用FastDeploy可以简单高效的在X86 CPU、NVIDIA GPU、飞腾CPU、ARM CPU、Intel GPU、昆仑、昇腾、瑞芯微、晶晨、算能等10+款硬件上对PaddleSeg模型进行快速部署，并且支持Paddle Inference、Paddle Lite、TensorRT、OpenVINO、ONNXRuntime、RKNPU2、SOPHGO等多种推理后端。

<div align="center">
    
<img src="https://user-images.githubusercontent.com/31974251/219546373-c02f24b7-2222-4ad4-9b43-42b8122b898f.png" >
    
</div>  

## 2. 语义分割模型部署  
<div id="语义分割模型部署"></div>  

### 2.1 硬件支持列表

|硬件类型|该硬件是否支持|使用指南|Python|C++|
|:---:|:---:|:---:|:---:|:---:|
|X86 CPU|✅|[链接](semantic_segmentation/cpu-gpu)|✅|✅|
|NVIDIA GPU|✅|[链接](semantic_segmentation/cpu-gpu)|✅|✅| 
|飞腾CPU|✅|[链接](semantic_segmentation/cpu-gpu)|✅|✅|
|ARM CPU|✅|[链接](semantic_segmentation/cpu-gpu)|✅|✅| 
|Intel GPU(集成显卡)|✅|[链接](semantic_segmentation/cpu-gpu)|✅|✅|  
|Intel GPU(独立显卡)|✅|[链接](semantic_segmentation/cpu-gpu)|✅|✅|    
|昆仑|✅|[链接](semantic_segmentation/kunlun)|✅|✅|
|昇腾|✅|[链接](semantic_segmentation/ascend)|✅|✅|
|瑞芯微|✅|[链接](semantic_segmentation/rockchip)|✅|✅|  
|晶晨|✅|[链接](semantic_segmentation/amlogic)|--|✅|✅|      
|算能|✅|[链接](semantic_segmentation/sophgo)|✅|✅|     

### 2.2. 详细使用文档
- X86 CPU
  - [部署模型准备](semantic_segmentation/cpu-gpu)  
  - [Python部署示例](semantic_segmentation/cpu-gpu/python/) 
  - [C++部署示例](semantic_segmentation/cpu-gpu/cpp/)
- NVIDIA GPU
  - [部署模型准备](semantic_segmentation/cpu-gpu)  
  - [Python部署示例](semantic_segmentation/cpu-gpu/python/) 
  - [C++部署示例](semantic_segmentation/cpu-gpu/cpp/)
- 飞腾CPU
  - [部署模型准备](semantic_segmentation/cpu-gpu)  
  - [Python部署示例](semantic_segmentation/cpu-gpu/python/) 
  - [C++部署示例](semantic_segmentation/cpu-gpu/cpp/)
- ARM CPU
  - [部署模型准备](semantic_segmentation/cpu-gpu)  
  - [Python部署示例](semantic_segmentation/cpu-gpu/python/) 
  - [C++部署示例](semantic_segmentation/cpu-gpu/cpp/)
- Intel GPU
  - [部署模型准备](semantic_segmentation/cpu-gpu)  
  - [Python部署示例](semantic_segmentation/cpu-gpu/python/) 
  - [C++部署示例](semantic_segmentation/cpu-gpu/cpp/)
- 昆仑 XPU
  - [部署模型准备](semantic_segmentation/kunlun)  
  - [Python部署示例](semantic_segmentation/kunlun/python/) 
  - [C++部署示例](semantic_segmentation/kunlun/cpp/)
- 昇腾 Ascend
  - [部署模型准备](semantic_segmentation/ascend)  
  - [Python部署示例](semantic_segmentation/ascend/python/) 
  - [C++部署示例](semantic_segmentation/ascend/cpp/)
- 瑞芯微 Rockchip
  - [部署模型准备](semantic_segmentation/rockchip/)  
  - [Python部署示例](semantic_segmentation/rockchip/rknpu2/) 
  - [C++部署示例](semantic_segmentation/rockchip/rknpu2/)
- 晶晨 Amlogic
  - [部署模型准备](semantic_segmentation/amlogic/a311d/)  
  - [C++部署示例](semantic_segmentation/amlogic/a311d/cpp/)    
- 算能 Sophgo
  - [部署模型准备](semantic_segmentation/sophgo/)  
  - [Python部署示例](semantic_segmentation/sophgo/python/) 
  - [C++部署示例](semantic_segmentation/sophgo/cpp/)  

### 2.3 更多部署方式

- [Android ARM CPU部署](semantic_segmentation/android)  
- [服务化Serving部署](semantic_segmentation/serving)  
- [web部署](semantic_segmentation/web)  
- [模型自动化压缩工具](semantic_segmentation/quantize)

## 3. Matting模型部署  
<div id="Matting模型部署"></div> 

### 3.1 硬件支持列表

|硬件类型|该硬件是否支持|使用指南|Python|C++|  
|:---:|:---:|:---:|:---:|:---:|   
|X86 CPU|✅|[链接](matting/cpu-gpu)|✅|✅|     
|NVIDIA GPU|✅|[链接](matting/cpu-gpu)|✅|✅|     
|飞腾CPU|✅|[链接](matting/cpu-gpu)|✅|✅|     
|ARM CPU|✅|[链接](matting/cpu-gpu)|✅|✅|     
|Intel GPU(集成显卡)|✅|[链接](matting/cpu-gpu)|✅|✅|     
|Intel GPU(独立显卡)|✅|[链接](matting/cpu-gpu)|✅|✅|    
|昆仑|✅|[链接](matting/kunlun)|✅|✅|     
|昇腾|✅|[链接](matting/ascend)|✅|✅|     

### 3.2 详细使用文档
- X86 CPU
  - [部署模型准备](matting/cpu-gpu)  
  - [Python部署示例](matting/cpu-gpu/python/) 
  - [C++部署示例](matting/cpu-gpu/cpp/)
- NVIDIA GPU
  - [部署模型准备](matting/cpu-gpu)  
  - [Python部署示例](matting/cpu-gpu/python/) 
  - [C++部署示例](matting/cpu-gpu/cpp/)
- 飞腾CPU
  - [部署模型准备](matting/cpu-gpu)  
  - [Python部署示例](matting/cpu-gpu/python/) 
  - [C++部署示例](matting/cpu-gpu/cpp/)
- ARM CPU
  - [部署模型准备](matting/cpu-gpu)  
  - [Python部署示例](matting/cpu-gpu/python/) 
  - [C++部署示例](matting/cpu-gpu/cpp/)
- Intel GPU
  - [部署模型准备](matting/cpu-gpu)  
  - [Python部署示例](matting/cpu-gpu/python/) 
  - [C++部署示例](cpu-gpu/cpp/)
- 昆仑 XPU
  - [部署模型准备](matting/kunlun)  
  - [Python部署示例](matting/kunlun/README.md) 
  - [C++部署示例](matting/kunlun/README.md)
- 昇腾 Ascend
  - [部署模型准备](matting/ascend)  
  - [Python部署示例](matting/ascend/README.md) 
  - [C++部署示例](matting/ascend/README.md)

## 4. 常见问题
<div id="常见问题"></div>   

遇到问题可查看常见问题集合，搜索FastDeploy issue，*或给FastDeploy提交[issue](https://github.com/PaddlePaddle/FastDeploy/issues)*:

[常见问题集合](https://github.com/PaddlePaddle/FastDeploy/tree/develop/docs/cn/faq)  
[FastDeploy issues](https://github.com/PaddlePaddle/FastDeploy/issues)  

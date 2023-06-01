[English](README.md) | 简体中文  

# PaddleSeg 量化模型部署-FastDeploy 

FastDeploy已支持部署量化模型,并提供一键模型自动化压缩的工具.
用户可以使用一键模型自动化压缩工具,自行对模型量化后部署, 也可以直接下载FastDeploy提供的量化模型进行部署.

## 1. FastDeploy一键模型自动化压缩工具  

FastDeploy 提供了一键模型自动化压缩工具, 能够简单地通过输入一个配置文件, 对模型进行量化.
详细教程请见: [一键模型自动化压缩工具](https://github.com/PaddlePaddle/FastDeploy/tree/develop/tools/common_tools/auto_compression)。**注意**: 推理量化后的分类模型仍然需要FP32模型文件夹下的deploy.yaml文件, 自行量化的模型文件夹内不包含此yaml文件, 用户从FP32模型文件夹下复制此yaml文件到量化后的模型文件夹内即可。

## 2. 量化完成的PaddleSeg模型  

用户也可以直接下载下表中的量化模型进行部署.(点击模型名字即可下载)

| 模型                 | 量化方式   |
|:----- | :-- |
| [PP-LiteSeg-T(STDC1)-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer_QAT_new.tar) |量化蒸馏训练 |

量化后模型的Benchmark比较，请参考[量化模型 Benchmark](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/quantize.md)

## 3. 支持部署量化模型的硬件  

FastDeploy 量化模型部署的过程大致都与FP32模型类似，只是模型量化与非量化的区别，如果硬件在量化模型部署过程有特殊处理，也会在文档中特别标明，因此量化模型部署可以参考如下硬件的链接

|硬件类型|该硬件是否支持|使用指南|Python|C++|
|:---:|:---:|:---:|:---:|:---:|
|X86 CPU|✅|[链接](cpu-gpu)|✅|✅|
|NVIDIA GPU|✅|[链接](cpu-gpu)|✅|✅| 
|飞腾CPU|✅|[链接](cpu-gpu)|✅|✅|
|ARM CPU|✅|[链接](cpu-gpu)|✅|✅| 
|Intel GPU(集成显卡)|✅|[链接](cpu-gpu)|✅|✅|  
|Intel GPU(独立显卡)|✅|[链接](cpu-gpu)|✅|✅|    
|昆仑|✅|[链接](kunlun)|✅|✅|
|昇腾|✅|[链接](ascend)|✅|✅|
|瑞芯微|✅|[链接](rockchip)|✅|✅|  
|晶晨|✅|[链接](amlogic)|--|✅|      
|算能|✅|[链接](sophgo)|✅|✅|       



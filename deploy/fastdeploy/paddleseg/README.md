# PaddleSeg高性能全场景模型部署方案—FastDeploy

## FastDeploy介绍

**[⚡️FastDeploy](https://github.com/PaddlePaddle/FastDeploy)**是一款**全场景**、**易用灵活**、**极致高效**的AI推理部署工具，支持**云边端**部署。使用FastDeploy可以简单高效的在X86 CPU、NVIDIA GPU、飞腾CPU、ARM CPU、Intel GPU、昆仑、昇腾、瑞芯微、晶晨、算能等10+款硬件上对PaddleSeg模型进行快速部署，并且支持Paddle Inference、Paddle Lite、TensorRT、OpenVINO、ONNXRuntime、RKNPU2、SOPHGO等多种推理后端。

<div align="center">
    
<img src="https://user-images.githubusercontent.com/54695910/213087733-7f2ea97b-baa4-4b0d-9b71-202ff6032a30.png" >
    
</div>  

## 硬件支持列表

|硬件类型|该硬件是否支持|使用指南|Python|C++|Linux|Windows|MacOS|Android|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|X86 CPU|✅|[链接](cpu-gpu)|✅|✅|✅|✅|✅|--|
|NVIDIA GPU|✅|[链接](cpu-gpu)|✅|✅|✅|✅|--|--|   
|飞腾CPU|✅|[链接](cpu-gpu)|✅|✅|✅|--|--|--|    
|ARM CPU|✅|[链接](cpu-gpu)|✅|✅|✅|--|✅|✅|    
|Intel GPU(集成显卡)|✅|[链接](cpu-gpu)|✅|✅|--|✅|--|--|   
|Intel GPU(独立显卡)|✅|[链接](cpu-gpu)|✅|✅|--|✅|--|--|     
|昆仑|✅|[链接](kunlun)|✅|✅|✅|--|--|--|     
|昇腾|✅|[链接](ascend)|✅|✅|✅|--|--|--|      
|瑞芯微|✅|[链接](rockchip)|✅|✅|✅|--|--|--|     
|晶晨|✅|[链接](amlogic)|--|✅|✅|--|--|--|      
|算能|✅|[链接](sophgo)|✅|✅|✅|--|--|--|      

## 更多部署方式

|Android ARM CPU部署|服务化Serving部署|web部署|模型自动化压缩工具|  
|:---:|:---:|:---:|:---:|  
|[使用指南](android)|[使用指南](serving)|[使用指南](web)|[使用指南](quantize)|

## 常见问题

遇到问题可查看常见问题集合文档或搜索FastDeploy issues，链接如下：

[常见问题集合](https://github.com/PaddlePaddle/FastDeploy/tree/develop/docs/cn/faq)

[FastDeploy issues](https://github.com/PaddlePaddle/FastDeploy/issues)

若以上方式都无法解决问题，欢迎给FastDeploy提交新的[issue](https://github.com/PaddlePaddle/FastDeploy/issues)

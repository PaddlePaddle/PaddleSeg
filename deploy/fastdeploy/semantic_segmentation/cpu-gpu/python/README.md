[English](README.md) | 简体中文  
# PaddleSeg CPU-GPU Python部署示例
本目录下提供`infer.py`快速完成PP-LiteSeg在CPU/GPU，以及GPU上通过Paddle-TensorRT加速部署的示例。执行如下脚本即可完成

## 1. 说明  
PaddleSeg支持利用FastDeploy在NVIDIA GPU、X86 CPU、飞腾CPU、ARM CPU、Intel GPU(独立显卡/集成显卡)硬件上快速部署Segmentation模型。

## 2. 部署环境准备  
在部署前，需确认软硬件环境，同时下载预编译部署库，参考[FastDeploy安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#FastDeploy预编译库安装)安装FastDeploy预编译库。

## 3. 部署模型准备
在部署前，请准备好您所需要运行的推理模型，你可以选择使用[预导出的推理模型](../README.md)或者[自行导出PaddleSeg部署模型](../README.md)，如果你部署的为**PP-Matting**、**PP-HumanMatting**以及**ModNet**请参考[Matting模型部署](../../../matting)。  

## 4. 运行部署示例
```bash
# 安装FastDpeloy python包（详细文档请参考`部署环境准备`）
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
conda config --add channels conda-forge && conda install cudatoolkit=11.2 cudnn=8.2

# 下载部署示例代码
git clone https://github.com/PaddlePaddle/PaddleSeg.git 
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到develop分支
# git checkout develop
cd PaddleSeg/deploy/fastdeploy/semantic_segmentation/cpp-gpu/python

# 下载Unet模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz
tar -xvf PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

# 运行部署示例
# CPU推理
python infer.py --model PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer --image cityscapes_demo.png --device cpu
# GPU推理
python infer.py --model PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer --image cityscapes_demo.png --device gpu
# GPU上使用Paddle-TensorRT推理 （注意：Paddle-TensorRT推理第一次运行，有序列化模型的操作，有一定耗时，需要耐心等待）
python infer.py --model PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer --image cityscapes_demo.png --device gpu --use_trt True
```

运行完成可视化结果如下图所示
<div  align="center">  
<img src="https://user-images.githubusercontent.com/16222477/191712880-91ae128d-247a-43e0-b1e3-cafae78431e0.jpg", width=512px, height=256px />
</div>

## 5. 部署示例选项说明  

|参数|含义|默认值
|---|---|---|  
|--model|指定模型文件夹所在的路径|None|
|--image|指定测试图片所在的路径|None|  
|--device|指定即将运行的硬件类型，支持的值为`[cpu, gpu]`，当设置为cpu时，可运行在x86 cpu/arm cpu等cpu上|cpu| 
|--use_trt|是否使用trt，该项只在device为gpu时有效|False|  

关于如何通过FastDeploy使用更多不同的推理后端，以及如何使用不同的硬件，请参考文档：[如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)

## 6. 更多指南
- [PaddleSeg python API文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/python/html/semantic_segmentation.html)
- [FastDeploy部署PaddleSeg模型概览](..)
- [PaddleSeg C++部署](../cpp)

## 7. 常见问题
- [如何将模型预测结果SegmentationResult转为numpy格式](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/vision_result_related_problems.md)
- [如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)
- [Intel GPU(独立显卡/集成显卡)的使用](https://github.com/PaddlePaddle/FastDeploy/blob/develop/tutorials/intel_gpu/README.md)
- [编译CPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/cpu.md)
- [编译GPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/gpu.md)
- [编译Jetson部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/jetson.md)

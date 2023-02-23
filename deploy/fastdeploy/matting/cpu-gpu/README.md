# PaddleSeg Matting模型高性能全场景部署方案-FastDeploy

## 1. 说明
PaddleSeg支持利用FastDeploy在NVIDIA GPU、X86 CPU、飞腾CPU、ARM CPU、Intel GPU(独立显卡/集成显卡)硬件上快速部署Matting模型

## 2. 使用预导出的模型列表
为了方便开发者的测试，下面提供了PP-Matting导出的各系列模型，开发者可直接下载使用。其中精度指标来源于PP-Matting中对各模型的介绍(未提供精度数据)，详情各参考PP-Matting中的说明。**注意**`deploy.yaml`文件记录导出模型的`input_shape`以及预处理信息，若不满足要求，用户可重新导出相关模型。

| 模型                                                               | 参数大小    | 精度    | 备注 |
|:---------------------------------------------------------------- |:----- |:----- | :------ |
| [PP-Matting-512](https://bj.bcebos.com/paddlehub/fastdeploy/PP-Matting-512.tgz) | 106MB | - |
| [PP-Matting-1024](https://bj.bcebos.com/paddlehub/fastdeploy/PP-Matting-1024.tgz) | 106MB | - |
| [PP-HumanMatting](https://bj.bcebos.com/paddlehub/fastdeploy/PPHumanMatting.tgz) | 247MB | - |
| [Modnet-ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/PPModnet_ResNet50_vd.tgz) | 355MB | - |
| [Modnet-MobileNetV2](https://bj.bcebos.com/paddlehub/fastdeploy/PPModnet_MobileNetV2.tgz) | 28MB | - |
| [Modnet-HRNet_w18](https://bj.bcebos.com/paddlehub/fastdeploy/PPModnet_HRNet_w18.tgz) | 51MB | - |

## 3. 自行导出PaddleSeg部署模型 
### 3.1 模型版本

支持[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/develop) 高于2.6版本的Matting模型，目前FastDeploy中测试过模型如下：  
- [PP-Matting系列模型](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/Matting)
- [PP-HumanMatting系列模型](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/Matting)
- [ModNet系列模型](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/Matting)

### 3.2 模型导出
PaddleSeg模型导出，请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/Matting)，**注意**：PaddleSeg导出的模型包含`model.pdmodel`、`model.pdiparams`和`deploy.yaml`三个文件，FastDeploy会从yaml文件中获取模型在推理时需要的预处理信息

## 4. 详细的部署示例  
- [Python部署](python)
- [C++部署](cpp)

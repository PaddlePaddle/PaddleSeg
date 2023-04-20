[English](README.md) | 简体中文  

# PaddleSeg 语义分割模瑞芯微NPU部署方案-FastDeploy

## 1. 说明   
本示例基于RV1126来介绍如何使用FastDeploy部署PaddleSeg模型，支持如下芯片的部署：  
- Rockchip RV1109
- Rockchip RV1126
- Rockchip RK1808

## 2. 预导出的量化推理模型
为了方便开发者的测试，下面提供了PaddleSeg导出的部分量化后的推理模型，开发者可直接下载使用。

| 模型                              | 参数文件大小    |输入Shape |  mIoU | mIoU (flip) | mIoU (ms+flip) |
|:---------------------------------------------------------------- |:----- |:----- | :----- | :----- | :----- |
| [PP-LiteSeg-T(STDC1)-cityscapes-without-argmax](https://bj.bcebos.com/fastdeploy/models/rk1/ppliteseg.tar.gz)| 31MB  | 1024x512 | 77.04% | 77.73% | 77.46% |
**注意**
- PaddleSeg量化模型包含`model.pdmodel`、`model.pdiparams`、`deploy.yaml`和`subgraph.txt`四个文件，FastDeploy会从yaml文件中获取模型在推理时需要的预处理信息，subgraph.txt是为了异构计算而存储的配置文件

## 3. 自行导出RV1126支持的INT8模型  

### 3.1 模型版本  
支持[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)高于2.6版本的Segmentation模型。目前FastDeploy测试过成功部署的模型:
- [PP-LiteSeg系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/configs/pp_liteseg/README.md)

### 3.2 模型导出
PaddleSeg模型导出，请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/docs/model_export_cn.md)，**注意**：PaddleSeg导出的模型包含`model.pdmodel`、`model.pdiparams`和`deploy.yaml`三个文件，FastDeploy会从yaml文件中获取模型在推理时需要的预处理信息

### 3.3 导出须知  
请参考[模型导出](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/docs/model_export_cn.md)中`output_op`参数的说明，获取您部署所需的模型，比如是否带`argmax`或`softmax`算子

### 3.4 转换为RV1126支持的INT8模型
瑞芯微RV1126仅支持INT8，将推理模型量化压缩为INT8模型，FastDeploy模型量化的方法及一键自动化压缩工具可以参考[模型量化](../../../quantize/README.md)

## 4. 详细的部署示例

目前，瑞芯微 RV1126 上只支持C++的部署。

- [C++部署](cpp)

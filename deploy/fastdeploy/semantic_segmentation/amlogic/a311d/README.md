[English](README.md) | 简体中文

# PaddleSeg 语义分割模型在晶晨NPU上的部署方案-FastDeploy

## 1. 说明  

晶晨A311D是一款先进的AI应用处理器。PaddleSeg支持通过FastDeploy在A311D上基于Paddle-Lite部署相关Segmentation模型。**注意**：需要注意的是，芯原（verisilicon）作为 IP 设计厂商，本身并不提供实体SoC产品，而是授权其 IP 给芯片厂商，如：晶晨（Amlogic），瑞芯微（Rockchip）等。因此本文是适用于被芯原授权了 NPU IP 的芯片产品。只要芯片产品没有大副修改芯原的底层库，则该芯片就可以使用本文档作为 Paddle Lite 推理部署的参考和教程。在本文中，晶晨 SoC 中的 NPU 和 瑞芯微 SoC 中的 NPU 统称为芯原 NPU。目前支持如下芯片的部署：
- Amlogic A311D
- Amlogic C308X
- Amlogic S905D3

本示例基于晶晨A311D来介绍如何使用FastDeploy部署PaddleSeg模型。

## 2. 使用预导出的模型列表  
| 模型                              | 参数文件大小    |输入Shape |  mIoU | mIoU (flip) | mIoU (ms+flip) |
|:---------------------------------------------------------------- |:----- |:----- | :----- | :----- | :----- |
| [PP-LiteSeg-T(STDC1)-cityscapes-without-argmax](https://bj.bcebos.com/fastdeploy/models/rk1/ppliteseg.tar.gz)| 31MB  | 1024x512 | 77.04% | 77.73% | 77.46% |
**注意**
- PaddleSeg量化模型包含`model.pdmodel`、`model.pdiparams`、`deploy.yaml`和`subgraph.txt`四个文件，FastDeploy会从yaml文件中获取模型在推理时需要的预处理信息，subgraph.txt是为了异构计算而存储的配置文件
- 若以上列表中无满足要求的模型，可参考下方教程自行导出适配A311D的模型

## 3. 自行导出晶晨A311D支持的PaddleSeg模型

### 3.1 模型版本
- 支持[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)高于2.6版本的Segmentation模型，目前FastDeploy测试过可在晶晨A311D成功部署的模型:   
- [PP-LiteSeg系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/configs/pp_liteseg/README.md)

### 3.2 PaddleSeg动态图模型导出为A311D支持的INT8模型
模型导出分为以下两步
1. PaddleSeg训练的动态图模型导出为推理静态图模型，请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/docs/model_export_cn.md)
晶晨A311D仅支持INT8
2. 将推理模型量化压缩为INT8模型，FastDeploy模型量化的方法及一键自动化压缩工具可以参考[模型量化](../../../quantize/README.md)

## 4. 详细部署示例

目前，A311D上只支持C++的部署。

- [C++部署](cpp)

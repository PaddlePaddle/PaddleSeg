[English](README.md) | 简体中文

# PaddleSeg利用FastDeploy基于RKNPU2部署Segmentation模型

RKNPU2 提供了一个高性能接口来访问 Rockchip NPU，支持如下硬件的部署
- RK3566/RK3568
- RK3588/RK3588S
- RV1103/RV1106

本示例基于 RV3588 来介绍如何使用 FastDeploy 部署 PaddleSeg 模型

## 模型版本说明

- [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)
>> **注意**：支持PaddleSeg高于2.6版本的Segmentation模型

目前FastDeploy使用RKNPU2推理PaddleSeg支持如下模型的部署:
- [U-Net系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/configs/unet/README.md)
- [PP-LiteSeg系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/configs/pp_liteseg/README.md)
- [PP-HumanSeg系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/contrib/PP-HumanSeg/README.md)
- [FCN系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/configs/fcn/README.md)
- [DeepLabV3系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/configs/deeplabv3/README.md)

## 准备PaddleSeg部署模型
PaddleSeg模型导出，请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/docs/model_export_cn.md)  

**注意**
- PaddleSeg导出的模型包含`model.pdmodel`、`model.pdiparams`和`deploy.yaml`三个文件，FastDeploy会从yaml文件中获取模型在推理时需要的预处理信息

## 下载预训练模型

为了方便开发者的测试，下面提供了PaddleSeg导出的部分模型
- without-argmax导出方式为：**不指定**`--input_shape`，**指定**`--output_op none`
- with-argmax导出方式为：**不指定**`--input_shape`，**指定**`--output_op argmax`

开发者可直接下载使用。

| 模型 | 参数文件大小 | 输入Shape  | mIoU   | mIoU (flip) | mIoU (ms+flip) |
|:----------------|:-------|:---------|:-------|:------------|:---------------|
| [Unet-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/Unet_cityscapes_without_argmax_infer.tgz)                                       | 52MB   | 1024x512 | 65.00% | 66.02%      | 66.89%         |
| [PP-LiteSeg-T(STDC1)-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer.tgz)          | 31MB   | 1024x512 | 77.04% | 77.73%      | 77.46%         |
| [PP-HumanSegV1-Lite(通用人像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV1_Lite_infer.tgz)                                      | 543KB  | 192x192  | 86.2%  | -           | -              |
| [PP-HumanSegV2-Lite(通用人像分割模型)](https://bj.bcebos.com/paddle2onnx/libs/PP_HumanSegV2_Lite_192x192_infer.tgz)                                  | 12MB   | 192x192  | 92.52% | -           | -              |
| [PP-HumanSegV2-Mobile(通用人像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV2_Mobile_192x192_infer.tgz)                          | 29MB   | 192x192  | 93.13% | -           | -              |
| [PP-HumanSegV1-Server(通用人像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV1_Server_infer.tgz)                                  | 103MB  | 512x512  | 96.47% | -           | -              |
| [Portait-PP-HumanSegV2_Lite(肖像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/Portrait_PP_HumanSegV2_Lite_256x144_infer.tgz)               | 3.6M   | 256x144  | 96.63% | -           | -              |
| [FCN-HRNet-W18-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/FCN_HRNet_W18_cityscapes_without_argmax_infer.tgz)                     | 37MB   | 1024x512 | 78.97% | 79.49%      | 79.74%         |
| [Deeplabv3-ResNet101-OS8-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/Deeplabv3_ResNet101_OS8_cityscapes_without_argmax_infer.tgz) | 150MB  | 1024x512 | 79.90% | 80.22%      | 80.47%         |

## 准备PaddleSeg部署模型以及转换模型
RKNPU部署模型前需要将Paddle模型转换成RKNN模型，具体步骤如下:
* PaddleSeg训练模型导出为推理模型，请参考[PaddleSeg模型导出说明](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/docs/model_export_cn.md)，也可以使用上表中的FastDeploy的预导出模型
* Paddle模型转换为ONNX模型，请参考[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)
* ONNX模型转换RKNN模型的过程，请参考[转换文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/rknpu2/export.md)进行转换。

上述步骤可参考以下具体示例

## 模型转换示例

* [PP-HumanSeg](./pp_humanseg.md)

## 详细部署文档
- [RKNN总体部署教程](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/rknpu2/rknpu2.md)
- [C++部署](cpp)
- [Python部署](python)

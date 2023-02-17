[English](README.md) | 简体中文
# PaddleSeg 使用 FastDeploy 服务化部署 Segmentation 模型
## FastDeploy 服务化部署介绍
在线推理作为企业或个人线上部署模型的最后一环，是工业界必不可少的环节，其中最重要的就是服务化推理框架。FastDeploy 目前提供两种服务化部署方式：simple_serving和fastdeploy_serving
- simple_serving：适用于只需要通过http等调用AI推理任务，没有高并发需求的场景。simple_serving基于Flask框架具有简单高效的特点，可以快速验证线上部署模型的可行性
- fastdeploy_serving：适用于高并发、高吞吐量请求的场景。基于Triton Inference Server框架，是一套可用于实际生产的完备且性能卓越的服务化部署框架

## 模型版本说明

- [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)
>> **注意**：支持PaddleSeg高于2.6版本的Segmentation模型

目前FastDeploy支持如下模型的部署

- [U-Net系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/configs/unet/README.md)
- [PP-LiteSeg系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/configs/pp_liteseg/README.md)
- [PP-HumanSeg系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/contrib/PP-HumanSeg/README.md)
- [FCN系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/configs/fcn/README.md)
- [DeepLabV3系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/configs/deeplabv3/README.md)
- [SegFormer系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/configs/segformer/README.md)

>>**注意** 如部署的为**PP-Matting**、**PP-HumanMatting**以及**ModNet**请参考[Matting模型部署](../../ppmatting)

## 准备PaddleSeg部署模型
PaddleSeg模型导出，请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/docs/model_export_cn.md)  

**注意**
- PaddleSeg导出的模型包含`model.pdmodel`、`model.pdiparams`和`deploy.yaml`三个文件，FastDeploy会从yaml文件中获取模型在推理时需要的预处理信息

## 预导出的推理模型

为了方便开发者的测试，下面提供了PaddleSeg导出的部分模型
- without-argmax导出方式为：**不指定**`--input_shape`，**指定**`--output_op none`
- with-argmax导出方式为：**不指定**`--input_shape`，**指定**`--output_op argmax`

开发者可直接下载使用。

| 模型                                                               | 参数文件大小    |输入Shape |  mIoU | mIoU (flip) | mIoU (ms+flip) |
|:---------------------------------------------------------------- |:----- |:----- | :----- | :----- | :----- |
| [Unet-cityscapes-with-argmax](https://bj.bcebos.com/paddlehub/fastdeploy/Unet_cityscapes_with_argmax_infer.tgz) \| [Unet-cityscapes-without-argmax](https://bj.bcebos.com/paddlehub/fastdeploy/Unet_cityscapes_without_argmax_infer.tgz)  | 52MB | 1024x512 | 65.00% | 66.02% | 66.89% |
| [PP-LiteSeg-B(STDC2)-cityscapes-with-argmax](https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer.tgz) \| [PP-LiteSeg-B(STDC2)-cityscapes-without-argmax](https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz) | 31MB  | 1024x512 | 79.04% |	79.52% | 79.85% |
|[PP-HumanSegV1-Lite-with-argmax(通用人像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/Portrait_PP_HumanSegV1_Lite_with_argmax_infer.tgz) \| [PP-HumanSegV1-Lite-without-argmax(通用人像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV1_Lite_infer.tgz) |  543KB | 192x192 | 86.2% | - | - |
|[PP-HumanSegV2-Lite-with-argmax(通用人像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV2_Lite_192x192_with_argmax_infer.tgz) \| [PP-HumanSegV2-Lite-without-argmax(通用人像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV2_Lite_192x192_infer.tgz) |  12MB | 192x192 | 92.52% | - | - |
| [PP-HumanSegV2-Mobile-with-argmax(通用人像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV2_Mobile_192x192_with_argmax_infer.tgz) \| [PP-HumanSegV2-Mobile-without-argmax(通用人像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV2_Mobile_192x192_infer.tgz) |  29MB | 192x192 | 93.13% | - | - |
|[PP-HumanSegV1-Server-with-argmax(通用人像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV1_Server_with_argmax_infer.tgz) \| [PP-HumanSegV1-Server-without-argmax(通用人像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV1_Server_infer.tgz) |  103MB | 512x512 | 96.47% | - | - |
| [Portait-PP-HumanSegV2-Lite-with-argmax(肖像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/Portrait_PP_HumanSegV2_Lite_256x144_with_argmax_infer.tgz) \| [Portait-PP-HumanSegV2-Lite-without-argmax(肖像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/Portrait_PP_HumanSegV2_Lite_256x144_infer.tgz) |  3.6M | 256x144 | 96.63% | - | - |
| [FCN-HRNet-W18-cityscapes-with-argmax](https://bj.bcebos.com/paddlehub/fastdeploy/FCN_HRNet_W18_cityscapes_with_argmax_infer.tgz) \| [FCN-HRNet-W18-cityscapes-without-argmax](https://bj.bcebos.com/paddlehub/fastdeploy/FCN_HRNet_W18_cityscapes_without_argmax_infer.tgz)(暂时不支持ONNXRuntime的GPU推理) |  37MB | 1024x512 | 78.97% | 79.49% | 79.74% |
| [Deeplabv3-ResNet101-OS8-cityscapes-with-argmax](https://bj.bcebos.com/paddlehub/fastdeploy/Deeplabv3_ResNet101_OS8_cityscapes_with_argmax_infer.tgz) \| [Deeplabv3-ResNet101-OS8-cityscapes-without-argmax](https://bj.bcebos.com/paddlehub/fastdeploy/Deeplabv3_ResNet101_OS8_cityscapes_without_argmax_infer.tgz) |  150MB | 1024x512 | 79.90% | 80.22% | 80.47% |
| [SegFormer_B0-cityscapes-with-argmax](https://bj.bcebos.com/paddlehub/fastdeploy/SegFormer_B0-cityscapes-with-argmax.tgz) \| [SegFormer_B0-cityscapes-without-argmax](https://bj.bcebos.com/paddlehub/fastdeploy/SegFormer_B0-cityscapes-without-argmax.tgz) |  15MB | 1024x1024 | 76.73% | 77.16% | - |

## 详细部署文档

- [fastdeploy serving](fastdeploy_serving)
- [simple serving](simple_serving)

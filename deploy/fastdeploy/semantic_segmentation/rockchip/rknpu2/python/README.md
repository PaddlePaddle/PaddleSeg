[English](README.md) | 简体中文
# PaddleSeg Python部署示例

在部署前，需确认以下步骤

- 1. 软硬件环境满足要求，RKNPU2环境部署等参考[FastDeploy环境要求](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/rknpu2/rknpu2.md)

【注意】如你部署的为**PP-Matting**、**PP-HumanMatting**以及**ModNet**请参考[Matting模型部署](../../../../../matting/)

本目录下提供`infer.py`快速完成PPHumanseg在RKNPU上部署的示例。执行如下脚本即可完成

```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/segmentation/paddleseg/python

# 下载图片
wget https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/images.zip
unzip images.zip

# 推理
python3 infer.py --model_file ./Portrait_PP_HumanSegV2_Lite_256x144_infer/Portrait_PP_HumanSegV2_Lite_256x144_infer_rk3588.rknn \
                --config_file ./Portrait_PP_HumanSegV2_Lite_256x144_infer/deploy.yaml \
                --image images/portrait_heng.jpg
```


## 注意事项
RKNPU上对模型的输入要求是使用NHWC格式，且图片归一化操作会在转RKNN模型时，内嵌到模型中，因此我们在使用FastDeploy部署时，需要先调用DisableNormalizeAndPermute(C++)或`disable_normalize_and_permute(Python)，在预处理阶段禁用归一化以及数据格式的转换。

## 快速链接

- [FastDeploy部署PaddleSeg模型概览](..)
- [PaddleSeg C++部署](../cpp)
- [转换PaddleSeg模型至RKNN模型文档](../README_CN.md#准备paddleseg部署模型以及转换模型)

## 常见问题
- [如何将模型预测结果SegmentationResult转为numpy格式](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/vision_result_related_problems.md)

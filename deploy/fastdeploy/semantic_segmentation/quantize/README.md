[English](README.md) | 简体中文
# PaddleSeg 量化模型部署
FastDeploy已支持部署量化模型,并提供一键模型自动化压缩的工具.
用户可以使用一键模型自动化压缩工具,自行对模型量化后部署, 也可以直接下载FastDeploy提供的量化模型进行部署.

## FastDeploy一键模型自动化压缩工具
FastDeploy 提供了一键模型自动化压缩工具, 能够简单地通过输入一个配置文件, 对模型进行量化.
详细教程请见: [一键模型自动化压缩工具](https://github.com/PaddlePaddle/FastDeploy/tree/develop/tools/common_tools/auto_compression)
>> **注意**: 推理量化后的分类模型仍然需要FP32模型文件夹下的deploy.yaml文件, 自行量化的模型文件夹内不包含此yaml文件, 用户从FP32模型文件夹下复制此yaml文件到量化后的模型文件夹内即可。

## 量化完成的PaddleSeg模型
用户也可以直接下载下表中的量化模型进行部署.(点击模型名字即可下载)

| 模型                 | 量化方式   |
|:----- | :-- |
| [PP-LiteSeg-T(STDC1)-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer_QAT_new.tar) |量化蒸馏训练 |

量化后模型的Benchmark比较，请参考[量化模型 Benchmark](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/quantize.md)

## 支持部署量化模型的硬件
FastDeploy 量化模型部署的过程大致都与FP32模型类似，只是模型量化与非量化的区别，如果硬件在量化模型部署过程有特殊处理，也会在文档中特别标明，因此量化模型部署可以参考如下硬件的链接

| 硬件支持列表 |  |   |   |
|:----- | :-- | :-- | :-- |
| [NVIDIA GPU](cpu-gpu) | [X86 CPU](cpu-gpu)| [飞腾CPU](cpu-gpu) | [ARM CPU](cpu-gpu) |
| [Intel GPU(独立显卡/集成显卡)](cpu-gpu) | [昆仑](kunlun) | [昇腾](ascend) | [瑞芯微](rockchip) |
| [晶晨](amlogic) | [算能](sophgo) |

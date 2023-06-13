[English](README.md) | 简体中文
# PP-Humanseg v1模型前端部署

## 模型版本说明

- [PP-HumanSeg Release/2.6](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/)


## 前端部署PP-Humanseg v1模型

PP-Humanseg v1模型web demo部署及使用参考[文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/examples/application/js/README_CN.md)


## PP-Humanseg v1 js接口

```
import * as humanSeg from "@paddle-js-models/humanseg";
# 模型加载与初始化
await humanSeg.load(Config);
# 人像分割
const res = humanSeg.getGrayValue(input)
# 提取人像与背景的二值图
humanSeg.drawMask(res)
# 用于替换背景的可视化函数
humanSeg.drawHumanSeg(res)
# 背景虚化
humanSeg.blurBackground(res)
```

**load()函数参数**
> * **Config**(dict): PP-Humanseg模型配置参数，默认为{modelpath : 'https://paddlejs.bj.bcebos.com/models/fuse/humanseg/humanseg_398x224_fuse_activation/model.json', mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5], enableLightModel: false}；modelPath为默认的PP-Humanseg js模型，mean，std分别为预处理的均值和标准差，enableLightModel为是否使用更轻量的模型。


**getGrayValue()函数参数**
> * **input**(HTMLImageElement | HTMLVideoElement | HTMLCanvasElement): 输入图像参数。

**drawMask()函数参数**
> * **seg_values**(number[]): 输入参数，一般是getGrayValue函数计算的结果作为输入

**blurBackground()函数参数**
> * **seg_values**(number[]): 输入参数，一般是getGrayValue函数计算的结果作为输入

**drawHumanSeg()函数参数**
> * **seg_values**(number[]): 输入参数，一般是getGrayValue函数计算的结果作为输入

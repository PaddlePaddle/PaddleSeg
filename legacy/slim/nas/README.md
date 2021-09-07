>运行该示例前请安装Paddle1.6或更高版本

# PaddleSeg神经网络搜索(NAS)示例

在阅读本教程前，请确保您已经了解过[PaddleSeg使用说明](../../docs/usage.md)等章节，以便对PaddleSeg有一定的了解

该文档介绍如何使用[PaddleSlim](https://paddlepaddle.github.io/PaddleSlim)对分割库中的模型进行搜索。

该教程中所示操作，如无特殊说明，均在`PaddleSeg/`路径下执行。

## 概述

我们选取Deeplab+mobilenetv2模型作为神经网络搜索示例，该示例使用[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)
辅助完成神经网络搜索实验，具体技术细节，请您参考[神经网络搜索策略](https://github.com/PaddlePaddle/PaddleSlim/blob/4670a79343c191b61a78e416826d122eea52a7ab/docs/zh_cn/tutorials/image_classification_nas_quick_start.ipynb)。


## 定义搜索空间
搜索实验中，我们采用了SANAS的方式进行搜索，本次实验会对网络模型中的通道数和卷积核尺寸进行搜索。
所以我们定义了如下搜索空间：
- head通道模块`head_num`：定义了MobilenetV2 head模块中通道数变化区间；
- inverse_res_block1-6`filter_num1-6`: 定义了inverse_res_block模块中通道数变化区间；
- inverse_res_block`repeat`：定义了MobilenetV2 inverse_res_block模块中unit的个数；
- inverse_res_block`multiply`：定义了MobilenetV2 inverse_res_block模块中expansion_factor变化区间；
- 卷积核尺寸`k_size`：定义了MobilenetV2中卷积和尺寸大小是3x3或者5x5。

根据定义的搜索空间各个区间，我们的搜索空间tokens共9位，变化区间在([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [7, 5, 8, 6, 2, 5, 8, 6, 2, 5, 8, 6, 2, 5, 10, 6, 2, 5, 10, 6, 2, 5, 12, 6, 2])范围内。  


初始化tokens为：[4, 4, 5, 1, 0, 4, 4, 1, 0, 4, 4, 3, 0, 4, 5, 2, 0, 4, 7, 2, 0, 4, 9, 0, 0]。

## 开始搜索
首先需要安装PaddleSlim，请参考[安装教程](https://paddlepaddle.github.io/PaddleSlim/#_2)。

配置paddleseg的config, 下面只展示nas相关的内容

```shell
SLIM:
    NAS_PORT: 23333 # 端口
    NAS_ADDRESS: "" # ip地址，作为server不用填写，作为client的时候需要填写server的ip
    NAS_SEARCH_STEPS: 100 # 搜索多少个结构
    NAS_START_EVAL_EPOCH: -1 # 第几个epoch开始对模型进行评估
    NAS_IS_SERVER: True # 是否为server
    NAS_SPACE_NAME: "MobileNetV2SpaceSeg" # 搜索空间
```

## 训练与评估
执行以下命令，边训练边评估
```shell
CUDA_VISIBLE_DEVICES=0 python -u ./slim/nas/train_nas.py --log_steps 10 --cfg configs/deeplabv3p_mobilenetv2_cityscapes.yaml --use_gpu \
SLIM.NAS_PORT 23333 \
SLIM.NAS_ADDRESS "" \
SLIM.NAS_SEARCH_STEPS 2 \
SLIM.NAS_START_EVAL_EPOCH -1 \
SLIM.NAS_IS_SERVER True \
SLIM.NAS_SPACE_NAME "MobileNetV2SpaceSeg" \
```


## FAQ
- 运行报错：`socket.error: [Errno 98] Address already in use`。

解决方法：当前端口被占用，请修改`SLIM.NAS_PORT`端口。

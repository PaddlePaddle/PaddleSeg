# 模型量化教程

## 1 概述

模型量化是一种常见的模型压缩方法，是使用整数替代浮点数进行存储和计算。

比如，模型量化将32bit浮点数转换成8bit整数，则模型存储空间可以减少4倍，同时整数运算替换浮点数运算，可以加快模型推理速度、降低计算内存。

PaddleSeg基于PaddleSlim，集成了量化训练（QAT）方法，特点如下：
* 概述：使用训练数据，在训练过程中更新权重，减小量化损失。
* 优点：量化模型的精度高；使用该量化模型预测，可以减少计算量、降低计算内存、减小模型大小。
* 缺点：易用性稍差，需要一定时间产出量化模型

## 2 量化模型精度和性能

测试环境：
* GPU: V100 32G
* CPU: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
* CUDA: 10.1
* cuDNN: 7.6
* TensorRT: 6.0.1.5
* Paddle: 2.1.1

测试方法:
1. 在GPU上使用TensorRT测试原始模型和量化模型
2. 使用cityspcaes的全量验证数据集(1024x2048)进行测试
3. 单GPU，Batchsize为1
4. 运行耗时为纯模型预测时间
5. 使用Paddle Inference的[Python API](../../deployment/inference/python_inference.md)测试，通过use_trt参数设置是否使用TRT，使用precision参数设置预测类型。

模型量化前后的精度和性能：

| 模型 | 类型 | mIoU |  耗时(s/img） | 量化加速比 |
| - | :-: | :-: | :-: | :-: |
| ANN_ResNet50_OS8 | FP32 | 0.7909  |  0.281  | - |
| ANN_ResNet50_OS8 | INT8 | 0.7906  |  0.195  | 30.6% |
| DANet_ResNet50_OS8 | FP32 | 0.8027  |  0.330  | - |
| DANet_ResNet50_OS8 | INT8 | 0.8039  |  0.266  | 19.4% |
| DeepLabV3P_ResNet50_OS8 | FP32 | 0.8036  | 0.206  |  - |  
| DeepLabV3P_ResNet50_OS8 | INT8 | 0.8044  | 0.083  | 59.7% |
| DNLNet_ResNet50_OS8 | FP32 | 0.7995  |  0.360  |  - |
| DNLNet_ResNet50_OS8 | INT8 | 0.7989  |  0.236  | 52.5% |
| EMANet_ResNet50_OS8 | FP32 |  0.7905  |  0.186  |  - |
| EMANet_ResNet50_OS8 | INT8 | 0.7939  |  0.106  | 43.0% |
| GCNet_ResNet50_OS8 | FP32 | 0.7950  |  0.228  |  - |
| GCNet_ResNet50_OS8 | INT8 | 0.7959  |  0.144  | 36.8% |
| PSPNet_ResNet50_OS8 | FP32 | 0.7883 | 0.324  |  - |
| PSPNet_ResNet50_OS8 | INT8 | 0.7915 | 0.223  | 32.1% |

## 3 示例

我们以一个示例来介绍如何产出和部署量化模型。
### 3.1 环境准备

请参考[安装文档](../../install.md)准备好PaddleSeg的基础环境。注意，量化功能要求PaddlePaddle版本>=2.2。

安装PaddleSlim。

```shell
git clone https://github.com/PaddlePaddle/PaddleSlim.git

# 切换到特定commit id
git reset --hard 15ef0c7dcee5a622787b7445f21ad9d1dea0a933

# 安装
python setup.py install
```

### 3.2 产出量化模型

#### 3.2.1 训练FP32模型

在产出量化模型之前，我们需要提前准备训练或者fintune好的FP32模型。

此处，我们选用视盘分割数据集和BiseNetV2模型，使用train.py从头开始训练模型。train.py输入参数的介绍，请参考[文档](../../train/train.md)。

在PaddleSeg目录下，执行如下脚本，会自动下载数据集进行训练。

```shell
# 设置1张可用的GPU卡
export CUDA_VISIBLE_DEVICES=0 
# windows下请执行以下命令
# set CUDA_VISIBLE_DEVICES=0

python train.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --do_eval \
       --use_vdl \
       --save_interval 250 \
       --save_dir output_fp32
```

训练结束后，精度最高的权重会保存到`output_fp32/best_model`目录下。

#### 3.2.2 使用量化训练方法产出量化模型

**1）产出量化模型**

基于训练好的FP32模型权重，使用`slim/quant/qat_train.py`进行量化训练。

qat_train.py和train.py的输入参数基本相似（如下）。注意，量化训练的学习率需要调小，使用`model_path`参数指定FP32模型的权重。

| 参数名              | 用途                                                         | 是否必选项 | 默认值           |
| ------------------- | ------------------------------------------------------------ | ---------- | ---------------- |
| config              | FP32模型的配置文件                                            | 是         |     -  | -                |
| model_path          | FP32模型的预训练权重                                        | 是  | - |
| iters               | 训练迭代次数                                                 | 否         | 配置文件中指定值 |
| batch_size          | 单卡batch size                                              | 否         | 配置文件中指定值 |
| learning_rate       | 初始学习率                                                   | 否         | 配置文件中指定值 |  
| save_dir            | 模型和visualdl日志文件的保存根路径                           | 否         | output           |
| num_workers         | 用于异步读取数据的进程数量， 大于等于1时开启子进程读取数据   | 否         | 0                |
| use_vdl             | 是否开启visualdl记录训练数据                                 | 否         | 否               |
| save_interval_iters | 模型保存的间隔步数                                           | 否         | 1000             |
| do_eval             | 是否在保存模型时启动评估, 启动时将会根据mIoU保存最佳模型至best_model | 否         | 否               |
| log_iters           | 打印日志的间隔步数                                           | 否         | 10               |
| resume_model        | 恢复训练模型路径，如：`output/iter_1000`                     | 否         | None  


执行如下命令，进行量化训练。量化训练结束后，精度最高的量化模型权重保存在`output_quant/best_model`目录下。

```shell
python slim/quant/qat_train.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --model_path output_fp32/best_model/model.pdparams \
       --learning_rate 0.001 \
       --do_eval \
       --use_vdl \
       --save_interval 250 \
       --save_dir output_quant
```

**2）测试量化模型**

如果需要，可以执行如下命令，使用`slim/quant/qat_val.py`脚本加载量化模型的权重，测试模型量化的精度。

```
python slim/quant/qat_val.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --model_path output_quant/best_model/model.pdparams
```

**3）导出量化预测模型**

基于训练好的量化模型权重，使用`slim/quant/qat_export.py`导出预测量化模型，脚本输入参数如下。

|参数名|用途|是否必选项|默认值|
|-|-|-|-|
|config|模型配置文件|是|-|
|save_dir|预测量化模型保存的文件夹|否|output|
|model_path|量化模型的权重|否|配置文件中指定值|
|with_softmax|在网络末端添加softmax算子。由于PaddleSeg组网默认返回logits，如果想要部署模型获取概率值，可以置为True|否|False|
|without_argmax|是否不在网络末端添加argmax算子。由于PaddleSeg组网默认返回logits，为部署模型可以直接获取预测结果，我们默认在网络末端添加argmax算子|否|False|

执行如下命令，导出预测量化模型保存在`output_quant_infer`目录。

```
python slim/quant/qat_export.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --model_path output_quant/best_model/model.pdparams \
       --save_dir output_quant_infer
```

### 3.3 部署量化模型

得到量化预测模型后，我们可以进行部署应用，请参考如下教程。


* [Paddle Inference Python部署](../../deployment/inference/python_inference.md)
* [Paddle Inference C++部署](../../deployment/inference/cpp_inference.md)
* [PaddleLite部署](../../deployment/lite/lite.md)

## 4 参考资料

* [PaddleSlim Github](https://github.com/PaddlePaddle/PaddleSlim)
* [PaddleSlim 文档](https://paddleslim.readthedocs.io/zh_CN/latest/)



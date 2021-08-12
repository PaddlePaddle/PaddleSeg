# 模型量化教程

模型量化是使用整数替代浮点数进行存储和计算的方法。举例而言，模型量化可以将32bit浮点数转换成8bit整数，则模型存储空间可以减少4倍，同时整数运算替换浮点数运算，可以加快模型推理速度、降低计算内存。

PaddleSeg基于PaddleSlim，集成了量化训练（QAT）方法，特点如下：
* 概述：使用大量训练数据，在训练过程中更新权重，减小量化损失。
* 注意事项：训练数据需要有Ground Truth。
* 优点：量化模型的精度高；使用该量化模型预测，可以减少计算量、降低计算内存、减小模型大小。
* 缺点：易用性稍差，需要一定时间产出量化模型

下面，本文以一个示例来介绍如何产出和部署量化模型。

## 1 环境准备

首先，请确保准备好PaddleSeg的基础环境。大家可以在PaddleSeg根目录执行如下命令，如果在`PaddleSeg/output`文件夹中出现预测结果，则证明安装成功。

```
python predict.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --model_path https://bj.bcebos.com/paddleseg/dygraph/optic_disc/bisenet_optic_disc_512x512_1k/model.pdparams\
       --image_path docs/images/optic_test_image.jpg \
       --save_dir output/result
```

然后，大家需要再安装最新版本的PaddleSlim。

```shell
pip install paddleslim -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2 产出量化模型

### 2.1 训练FP32模型

在产出量化模型之前，我们需要提前准备训练或者fintune好的FP32模型。

此处，我们选用视盘分割数据集和BiseNetV2模型，从头开始训练模型。

在PaddleSeg目录下，执行如下脚本，会自动下载数据集进行训练。训练结束后，精度最高的权重会保存到`output_fp32/best_model`目录下。

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

### 2.2 使用量化训练方法产出量化模型

**训练量化模型**

基于2.1步骤中训练好的FP32模型权重，执行如下命令，使用`slim/quant/qat_train.py`脚本进行量化训练。

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

上述脚本的输入参数和常规训练相似，复用2.1步骤的config文件，使用`model_path`参数指定FP32模型的权重，初始学习率相应调小。

训练结束后，精度最高的量化模型权重会保存到`output_quant/best_model`目录下。

**测试量化模型**

执行如下命令，使用`slim/quant/qat_val.py`脚本加载量化模型的权重，测试模型量化的精度。

```
python slim/quant/qat_val.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --model_path output_quant/best_model/model.pdparams
```

**导出量化预测模型**

基于此前训练好的量化模型权重，执行如下命令，使用`slim/quant/qat_export.py`导出预测量化模型，保存在`output_quant_infer`目录下。

```
python slim/quant/qat_export.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --model_path output_quant/best_model/model.pdparams \
       --save_dir output_quant_infer
```

## 3 部署

得到量化预测模型后，我们可以直接进行部署应用，相关教程请参考:
* [Paddle Inference部署](../../deployment/inference/inference.md)
* [PaddleLite部署](../../deployment/lite/lite.md)

## 4 量化加速比

测试环境：
* GPU: V100
* CPU: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
* CUDA: 10.2
* cuDNN: 7.6
* TensorRT: 6.0.1.5

测试方法:
1. 运行耗时为纯模型预测时间，测试图片cityspcaes(1024x2048)
2. 预测10次作为热启动，连续预测50次取平均得到预测时间
3. 使用GPU + TensorRT测试

|模型|未量化运行耗时(ms)|量化运行耗时(ms)|加速比|
|-|-|-|-|
|deeplabv3_resnet50_os8|204.2|150.1|26.49%|
|deeplabv3p_resnet50_os8|147.2|89.5|39.20%|
|gcnet_resnet50_os8|201.8|126.1|37.51%|
|pspnet_resnet50_os8|266.8|206.8|22.49%|  

## 5 参考资料

* [PaddleSlim Github](https://github.com/PaddlePaddle/PaddleSlim)
* [PaddleSlim 文档](https://paddleslim.readthedocs.io/zh_CN/latest/)

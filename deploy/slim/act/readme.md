# 语义分割模型自动压缩示例

目录：
- [1.简介](#1简介)
- [2.Benchmark](#2Benchmark)
- [3.开始自动压缩](#自动压缩流程)
  - [3.1 环境准备](#31-准备环境)
  - [3.2 准备数据集](#32-准备数据集)
  - [3.3 准备预测模型](#33-准备预测模型)
  - [3.4 自动压缩并产出模型](#34-自动压缩并产出模型)
- [4.预测部署](#4预测部署)
- [5.FAQ](5FAQ)

## 1.简介

本示例将以语义分割模型[PP-Liteseg](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/configs/pp_liteseg)为例，介绍如何使用PaddleSlim中的ACT压缩工具型进行自动压缩。本示例使用的自动压缩策略为量化蒸馏训练和离线量化。

## 2.Benchmark

| 模型 | 策略  | Total IoU (%) | CPU耗时(ms)<br>thread=10<br>mkldnn=on| Nvidia GPU耗时(ms)<br>TRT=on| 配置文件 | Inference模型  |
|:-----:|:-----:|:----------:|:---------:| :------:|:------:|:------:|
| OCRNet_HRNetW48 |Baseline |todo| todo| todo|todo|todo|
| OCRNet_HRNetW48 | 量化蒸馏训练 |todo| todo| todo|todo|todo|
| SegFormer-B0  |Baseline |todo| todo| todo|todo|todo|
| SegFormer-B0  |量化蒸馏训练 |todo| todo| todo|todo|todo|
| PP-LiteSeg-Tiny  |Baseline | 77.04 | 640.72 | 1262.0ms | - |[model](https://paddleseg.bj.bcebos.com/deploy/slim_act/ppliteseg/liteseg_tiny_scale1.0.zip)|
| PP-LiteSeg-Tiny  |量化蒸馏训练 | 77.14 | 450.19 | 1212.7ms | [config](./configs/ppliteseg/ppliteseg_qat.yaml)|[model](https://paddleseg.bj.bcebos.com/deploy/slim_act/ppliteseg/save_quant_model_qat.zip)|
| PP-MobileSeg-Base  |Baseline |todo| todo| todo|todo|todo|
| PP-MobileSeg-Base  |量化蒸馏训练 |todo| todo| todo|todo|todo|

- CPU测试环境：
  - Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
  - cpu thread: 10


- Nvidia GPU测试环境：

  - 硬件：NVIDIA Tesla V100 单卡
  - 软件：CUDA 10.2, cuDNN 7.6.5, TensorRT 8.0
  - 测试配置：batch_size: 32

下面将以开源数据集为例介绍如何对PP-Liteseg进行自动压缩。

## 3. 自动压缩流程

#### 3.1 准备环境

- PaddlePaddle == 2.5 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim == 2.5
- PaddleSeg == develop

安装paddlepaddle：
```shell
# CPU
python -m pip install paddlepaddle==2.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
# GPU 以Ubuntu、CUDA 10.2为例
python -m pip install paddlepaddle-gpu==2.5.1.post102 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

安装paddleslim 2.5：
```shell
git clone https://github.com/PaddlePaddle/PaddleSlim.git && cd PaddleSlim/
git fetch origin release/2.5
git checkout release/2.5
python setup.py install  
```

安装paddleseg develop和对应包：
```shell
cd ..
git clone https://github.com/PaddlePaddle/PaddleSeg.git
cd PaddleSeg/
git fetch develop
git checkout FETCH_HEAD
git checkout -b develop
python setup.py install
```

#### 3.2 准备数据集

开发者可下载开源数据集 (如[Cityscapes](https://bj.bcebos.com/v1/paddle-slim-models/data/mini_cityscapes/mini_cityscapes.tar)) 或自定义语义分割数据集。请参考[PaddleSeg数据准备文档](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/docs/data/marker/marker_cn.md)来检查、对齐数据格式即可。

本示例使用示例开源数据集 Cityscapes 数据集为例介绍如何对PP-Liteseg-Tiny进行自动压缩。示例数据集仅用于快速跑通自动压缩流程，并不能复现出 benckmark 表中的压缩效果。

- 示例数据集: cityscapes数据集的一个子集，用于快速跑通压缩和推理流程，不能用该数据集复现 benchmark 表中的压缩效果。[下载链接](https://bj.bcebos.com/v1/paddle-slim-models/data/mini_cityscapes/mini_cityscapes.tar)

准备好数据后，需要放入到`deploy/slim/act/data/cityscapes`目录下。

#### 3.3 准备预测模型

- 通过下面的指令可以对ppliteseg-tiny的模型进行导出，其他的模型导出可以参照[导出指南](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/docs/model_export_cn.md)：

```shell
cd PaddleSeg/
wget https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc1_cityscapes_1024x512_scale1.0_160k/model.pdparams

python tools/export.py --config configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k.yml --model_path model.pdparams  --save_dir liteseg_tiny_scale1.0
```

- 导出模型后，需要指定模型路径到 model_filename 和 params_filename。

预测模型的格式为：`model.pdmodel` 和 `model.pdiparams`两个，带`pdmodel`的是模型文件，带`pdiparams`后缀的是权重文件。

注：其他像`__model__`和`__params__`分别对应`model.pdmodel` 和 `model.pdiparams`文件。

#### 3.4 自动压缩并产出模型

自动压缩示例通过run_seg.py脚本启动，会使用接口 ```paddleslim.auto_compression.AutoCompression``` 对模型进行自动压缩。首先要配置config文件中模型路径、数据集路径、蒸馏、量化、稀疏化和训练等部分的参数，配置完成后便可对模型进行非结构化稀疏、蒸馏和量化、蒸馏。


- 自行配置量化参数进行量化蒸馏训练，配置参数含义详见[自动压缩超参文档](https://github.com/PaddlePaddle/PaddleSlim/blob/27dafe1c722476f1b16879f7045e9215b6f37559/demo/auto_compression/hyperparameter_tutorial.md)。具体命令如下所示：

```shell
# 单卡启动
export CUDA_VISIBLE_DEVICES=0
cd PaddleSeg/deploy/slim/act/
python run_seg.py --act_config_path='./configs/ppliteseg/ppliteseg_qat.yaml' --save_dir='./save_quant_model_qat' --config_path ../../../configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale1.0_160k.yml

# 多卡启动
export CUDA_VISIBLE_DEVICES=0,1
cd PaddleSeg/deploy/slim/act/
python -m paddle.distributed.launch run_seg.py --act_config_path='./configs/ppliteseg/ppliteseg_qat.yaml' --save_dir='./save_quant_model_qat' --config_path ../../../configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale1.0_160k.yml
```

压缩完成后会在`save_dir`中产出压缩好的预测模型，可直接预测部署。


## 4.预测部署

#### 4.1 Paddle Inference 验证性能

输出的量化模型也是静态图模型，静态图模型在GPU上可以使用TensorRT进行加速，在CPU上可以使用MKLDNN进行加速。预测可以参考[预测文档](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/docs/deployment/inference/python_inference_cn.md)


以下字段用于配置预测参数：

| 参数名 | 含义 |
|:------:|:------:|
| model_path | inference 模型文件所在目录，该目录下需要有文件 .pdmodel 和 .pdiparams 两个文件 |
| model_filename | inference_model_dir文件夹下的模型文件名称 |
| params_filename | inference_model_dir文件夹下的参数文件名称 |
| dataset | 选择数据集的类型，可选：`human`, `cityscape`。  |
| dataset_config | 数据集配置的config  |
| image_file | 待测试单张图片的路径，如果设置image_file，则dataset_config将无效。   |
| device | 预测时的设备，可选：`CPU`, `GPU`。  |
| use_trt | 是否使用 TesorRT 预测引擎，在device为```GPU```时生效。   |
| use_mkldnn | 是否启用```MKL-DNN```加速库，注意```use_mkldnn```，在device为```CPU```时生效。  |
| cpu_threads | CPU预测时，使用CPU线程数量，默认10  |
| precision | 预测时精度，可选：`fp32`, `fp16`, `int8`。 |


- TensorRT预测：

环境配置：
1. 如果使用 TesorRT 预测引擎，需安装 ```WITH_TRT=ON``` 的Paddle，上述paddle下载的2.5满足打开TensorRT编译的要求。
2. 使用TensorRT预测需要进一步安装TensorRT，安装TensorRT的方式参考[TensorRT安装说明](../../../docs/deployment/installtrt.md)。


准备好预测模型，并且修改dataset_config中数据集路径为正确的路径后，启动测试：

```shell
cd PaddleSeg/deploy/slim/act/
python test_seg.py \
      --model_path=save_quant_model_qat \
      --dataset='cityscape' \
      --config=../../../configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale1.0_160k.yml \
      --precision=int8 \
      --use_trt True
```

- MKLDNN预测：

```shell
cd PaddleSeg/deploy/slim/act/
python test_seg.py \
      --model_path=save_quant_model_qat \
      --dataset='cityscape' \
      --config=../../../configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale1.0_160k.yml \
      --device=CPU \
      --use_mkldnn=True \
      --precision=int8 \
      --cpu_threads=10
```

#### 4.2 Paddle Inference 测试单张图片

基于量化模型测试单张图片：

```shell
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

cd PaddleSeg/deploy/slim/act/
python test_seg.py \
      --model_path=liteseg_tiny_scale1.0 \
      --dataset='cityscape' \
      --image_file=cityscapes_demo.png \
      --use_trt=True \
      --precision=fp32 \
      --save_file res_qat_fp32.png
```

```shell
cd PaddleSeg/deploy/slim/act/

wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

python test_seg.py \
      --model_path=save_quant_model_qat \
      --dataset='cityscape' \
       --image_file=cityscapes_demo.png \
      --use_trt=True \
      --precision=int8 \
      --save_file res_qat_int8.png
```

<table><tbody>

<tr>
<td>
原始图片
</td>
<td>
<img src="https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png" width="340" height="200">
</td>
</tr>

<tr>
<td>
FP32推理结果
</td>
<td>
<img src="https://github.com/PaddlePaddle/PaddleSeg/assets/34859558/76ef63a8-09de-455d-a451-cdaf75bde2dc" width="340" height="200">
</td>
</tr>

<tr>
<td>
Int8推理结果
</td>
<td>
<img src="https://github.com/PaddlePaddle/PaddleSeg/assets/34859558/d8d29a38-e024-4b81-ab5a-74bf573b3349" width="340" height="200">
</td>
</tr>

</tbody></table>

### 4.3 更多部署教程

- [Paddle Inference Python部署](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/docs/deployment/inference/python_inference.md)
- [Paddle Inference C++部署](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/docs/deployment/inference/cpp_inference.md)
- [Paddle Lite部署](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/docs/deployment/lite/lite.md)

## 5.FAQ
### 1. paddleslim 和 paddleseg 存在opencv的版本差异？

**A**：去除Paddleslim中requirements.txt的opencv版本限制后重新安装

### 2. 报错：Distill_node_pair config wrong, the length need to be an even number ？
<td>
<img src="https://github.com/PaddlePaddle/PaddleOCR/assets/34859558/c4c7624a-a31c-4278-af4e-c0a2a34fa22b" width="200" height="340">
</td>

**A**：蒸馏配置中的node需要设置成网络的输出节点。1. 使用netron打开静态图模型model.pdmodel，2. 修改node为最后一层卷积的输出名字。
<td>
<img src="https://github.com/PaddlePaddle/PaddleOCR/assets/34859558/b714040a-eec1-43df-af11-a233bd4cb59b" width="2340" height="100">
</td>

<img width="1000" alt="e589cdafd43796aed4c1b11c6828fefd" src="https://github.com/PaddlePaddle/PaddleOCR/assets/34859558/c75c0749-898b-4187-a9f9-363cfe92b1a4">


### 3. 量化蒸馏训练精度很低？
![2d916558811eb5f1bbb388025ddda21c](https://github.com/PaddlePaddle/PaddleOCR/assets/34859558/cc9bcc26-1568-4ab9-96f3-ff181486637c)

**A**：去除量化训练的输出结果，重新运行一次，这是由于网络训练到局部极值点导致。

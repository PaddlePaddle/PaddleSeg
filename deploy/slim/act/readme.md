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

本示例将以语义分割模型[PP-Liteseg](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/configs/pp_liteseg)为例，介绍如何使用PaddleSlim中的ACT压缩工具型进行自动压缩。本示例使用的自动压缩策略为量化蒸馏训练。

## 2.Benchmark

| 模型 | 策略  | Total IoU (%) | CPU耗时(ms)<br>thread=10<br>mkldnn=on| Nvidia GPU耗时(ms)<br>TRT=on| 配置文件 | Inference模型  |
|:-----:|:-----:|:----------:|:---------:| :------:|:------:|:------:|
| OCRNet_HRNetW48 |Baseline |82.15| **4332.2** | **154.9** | - | [mode](https://paddleseg.bj.bcebos.com/deploy/slim_act/ocrnet/ocrnet_export.zip)|
| OCRNet_HRNetW48 | 量化蒸馏训练 |82.03| **3728.7** | **59.8**|[config](configs/ocrnet/ocrnet_hrnetw48_qat.yaml)| [model](https://paddleseg.bj.bcebos.com/deploy/slim_act/ocrnet/ocrnet_qat.zip) |
| SegFormer-B0*  |Baseline | 75.27| 285.4| 34.3 |-| [model](https://paddleseg.bj.bcebos.com/deploy/slim_act/segformer/segformer_b0_export.zip) |
| SegFormer-B0*  |量化蒸馏训练 | 75.22 | 284.1| 35.7|[config](configs/segformer/segformer_b0_qat.yaml)| [model](https://paddleseg.bj.bcebos.com/deploy/slim_act/segformer/segformer_qat.zip) |
| PP-LiteSeg-Tiny  |Baseline | 77.04 | 640.72 | **11.9** | - |[model](https://paddleseg.bj.bcebos.com/deploy/slim_act/ppliteseg/liteseg_tiny_scale1.0.zip)|
| PP-LiteSeg-Tiny  |量化蒸馏训练 | 77.14 | 450.19 | **7.5** | [config](./configs/ppliteseg/ppliteseg_qat.yaml)|[model](https://paddleseg.bj.bcebos.com/deploy/slim_act/ppliteseg/save_quant_model_qat.zip)|
| PP-MobileSeg-Base  |Baseline |41.55| **311.1** | **17.8** | - | [model](https://paddleseg.bj.bcebos.com/deploy/slim_act/ppmobileseg/ppmobileseg_base_ade_export.zip) |
| PP-MobileSeg-Base  |量化蒸馏训练 |39.08| **303.6** | **16.2**| [config](configs/ppmobileseg/ppmobileseg_qat.yml)| [model](https://paddleseg.bj.bcebos.com/deploy/slim_act/ppmobileseg/ppmobileseg_base_ade.zip)|

* SegFormer-B0 is tested on CPU under deleted gpu_cpu_map_matmul_v2_to_mul_pass because it will raise an error.
* PP-MobileSeg-Base is tested on ADE20K dataset, while others are tested on cityscapes.

- CPU测试环境：
  - Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz
  - cpu thread: 10


- Nvidia GPU测试环境：

  - 硬件：NVIDIA Tesla V100 单卡
  - 软件：CUDA 11.2, cudnn 8.1.0, TensorRT-8.0.3.4
  - 测试配置：batch_size: 4

- 测速要求：
  - 批量测试取平均：单张图片上测速时间会有浮动，因此测速需要跑10遍warmup，再跑100次取平均。现有test_seg的批量测试已经集成该功能。
  - 确认TRT加速：检查下int8模型是否开启了trt int8模式，确认预测中有没有trt pass，比如看下有无这个pass：trt_delete_weight_dequant_linear_op_pass
  - 确认是否开启了动态shape的功能？如果是，则需要跑两遍，第一次会在采集shape大小，需要以第二次的时间为准，

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
pip install paddleslim@git+https://gitee.com/paddlepaddle/PaddleSlim.git@release/2.5
```

安装paddleseg develop和对应包：
```shell
cd ..
git clone https://github.com/PaddlePaddle/PaddleSeg.git -b develop
cd PaddleSeg/
python setup.py install
```

#### 3.2 准备数据集

1. 开发者可下载开源数据集 (如[Cityscapes](https://bj.bcebos.com/v1/paddle-slim-models/data/mini_cityscapes/mini_cityscapes.tar)) 或参考[PaddleSeg数据准备文档](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/docs/data/marker/marker_cn.md#4%E6%95%B0%E6%8D%AE%E9%9B%86%E6%96%87%E4%BB%B6%E6%95%B4%E7%90%86)来自定义语义分割数据集。

2. 本示例使用示例开源数据集 Cityscapes 数据集为例介绍如何对PP-Liteseg-Tiny进行自动压缩。示例数据集仅用于快速跑通自动压缩流程，并不能复现出 benckmark 表中的压缩效果。[下载链接](https://bj.bcebos.com/v1/paddle-slim-models/data/mini_cityscapes/mini_cityscapes.tar)

3. 准备好数据后，需要放入到`deploy/slim/act/data/cityscapes`目录下。

#### 3.3 准备预测模型

- 通过下面的指令可以对ppliteseg-tiny的模型进行导出，其他的模型导出可以参照[导出指南](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/docs/model_export_cn.md)：

```shell
cd PaddleSeg/
wget https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc1_cityscapes_1024x512_scale1.0_160k/model.pdparams

python tools/export.py --config configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k.yml --model_path model.pdparams  --save_dir ppliteseg_tiny_scale1.0_export
```

- 导出模型后，需要指定模型路径到配置文件中的 model_filename 和 params_filename。
- 预测模型的格式为：`model.pdmodel` 和 `model.pdiparams`两个，带`pdmodel`的是模型文件，带`pdiparams`后缀的是权重文件。


#### 3.4 自动压缩并产出模型

自动压缩示例通过run_seg.py脚本启动，会使用接口 ```paddleslim.auto_compression.AutoCompression``` 对模型进行自动压缩。首先要配置config文件中模型路径、数据集路径、蒸馏、量化、稀疏化和训练等部分的参数，配置完成后便可对模型进行非结构化稀疏、蒸馏和量化、蒸馏。


- 自行配置量化参数进行量化蒸馏训练，配置参数含义详见[自动压缩超参文档](https://github.com/PaddlePaddle/PaddleSlim/blob/27dafe1c722476f1b16879f7045e9215b6f37559/demo/auto_compression/hyperparameter_tutorial.md)。具体命令如下所示：

```shell
# 单卡启动
export CUDA_VISIBLE_DEVICES=0
cd PaddleSeg/deploy/slim/act/
python run_seg.py \
      --act_config_path='./configs/ppliteseg/ppliteseg_qat.yaml' \
      --save_dir='./save_quant_model_qat'  \
      --config_path="configs/datasets/pp_liteseg_1.0_data.yml"

# 多卡启动
export CUDA_VISIBLE_DEVICES=0,1
cd PaddleSeg/deploy/slim/act/
python -m paddle.distributed.launch run_seg.py \
      --act_config_path='./configs/ppliteseg/ppliteseg_qat.yaml' \
      --save_dir='./save_quant_model_qat'  \
      --config_path="configs/datasets/pp_liteseg_1.0_data.yml"
```

压缩完成后会在`save_dir`中产出压缩好的预测模型，可直接预测部署。


## 4.预测部署

#### 4.1 Paddle Inference 验证性能

输出的量化模型也是静态图模型，静态图模型在GPU上可以使用TensorRT进行加速，在CPU上可以使用MKLDNN进行加速。预测可以参考[预测文档](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/docs/deployment/inference/python_inference_cn.md)。

TensorRT预测环境配置：
1. 如果使用 TesorRT 预测引擎，需安装 ```WITH_TRT=ON``` 的Paddle，上述paddle下载的2.5满足打开TensorRT编译的要求。
2. 使用TensorRT预测需要进一步安装TensorRT，安装TensorRT的方式参考[TensorRT安装说明](../../../docs/deployment/installtrt.md)。

以下字段用于配置预测参数：

| 参数名 | 含义 |
|:------:|:------:|
| model_path | inference 模型文件所在目录，该目录下需要有文件 .pdmodel 和 .pdiparams 两个文件 |
| model_filename | inference_model_dir文件夹下的模型文件名称 |
| params_filename | inference_model_dir文件夹下的参数文件名称 |
| dataset | 选择数据集的类型，可选：`human`, `cityscapes`, `ade`。  |
| dataset_config | 数据集配置的config  |
| image_file | 待测试单张图片的路径，如果设置image_file，则dataset_config将无效。   |
| device | 预测时的设备，可选：`CPU`, `GPU`。  |
| use_trt | 是否使用 TesorRT 预测引擎，在device为```GPU```时生效。   |
| use_mkldnn | 是否启用```MKL-DNN```加速库，注意```use_mkldnn```，在device为```CPU```时生效。  |
| cpu_threads | CPU预测时，使用CPU线程数量，默认10  |
| precision | 预测时精度，可选：`fp32`, `fp16`, `int8`。 |


准备好预测模型，并且修改dataset_config中数据集路径为正确的路径后，启动测试：

##### 4.1.1 基于压缩模型进行基于GPU的批量测试：

```shell
cd PaddleSeg/deploy/slim/act/
python test_seg.py \
      --model_path=save_quant_model_qat \
      --dataset='cityscapes' \
      --config="configs/datasets/pp_liteseg_1.0_data.yml" \
      --precision=int8 \
      --use_trt=True
```
预期结果：

<img src="https://github.com/PaddlePaddle/PaddleSlim/assets/34859558/75119e54-28c1-4b3c-8c91-ab5ba6afb677" width="600" height="150">


##### 4.1.2 基于压缩前模型进行基于GPU的批量测试：

```shell
cd PaddleSeg/deploy/slim/act/
python test_seg.py \
      --model_path=ppliteseg_tiny_scale1.0_export/ \
      --dataset='cityscapes' \
      --config="configs/datasets/pp_liteseg_1.0_data.yml" \
      --precision=fp32 \
      --use_trt=True
```
预期结果：

<img src="https://github.com/PaddlePaddle/PaddleSlim/assets/34859558/d46c911f-2880-41ad-b0cc-c092eb9fbb05" width="600" height="150">


##### 4.1.3 基于压缩模型进行基于CPU的批量测试：

- MKLDNN预测：

```shell
cd PaddleSeg/deploy/slim/act/
python test_seg.py \
      --model_path=save_quant_model_qat \
      --dataset='cityscapes' \
      --config="configs/datasets/pp_liteseg_1.0_data.yml" \
      --device=CPU \
      --use_mkldnn=True \
      --precision=int8 \
      --cpu_threads=10
```

#### 4.2 Paddle Inference 测试单张图片

##### 4.2.1 基于压缩前模型测试单张图片：

```shell
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

cd PaddleSeg/deploy/slim/act/
python test_seg.py \
      --model_path=ppliteseg_tiny_scale1.0_export \
      --dataset='cityscapes' \
      --image_file=cityscapes_demo.png \
      --use_trt=True \
      --precision=fp32 \
      --save_file res_qat_fp32.png
```
预期结果：
<img src="https://github.com/PaddlePaddle/PaddleSlim/assets/34859558/dc93b323-60e3-48b9-aac2-e2a165ca6a3c" width="800" height="600">

##### 4.2.2  基于压缩模型测试单张图片：

```shell
cd PaddleSeg/deploy/slim/act/

wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

python test_seg.py \
      --model_path=save_quant_model_qat \
      --dataset='cityscapes' \
      --image_file=cityscapes_demo.png \
      --use_trt=True \
      --precision=int8 \
      --save_file res_qat_int8.png
```

预期结果：

<img src="https://github.com/PaddlePaddle/PaddleSlim/assets/34859558/c67ed087-20e3-4c47-aedd-69a4bb6dff5a" width="800" height="700">

#####  4.2.3 图片结果对比

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

**A**：去除Paddleslim中requirements.txt的opencv版本限制后重新安装。

### 2. 报错：Distill_node_pair config wrong, the length need to be an even number ？
<td>
<img src="https://github.com/PaddlePaddle/PaddleSlim/assets/34859558/c9687940-f08f-4eac-bccf-1b98517de771" width="800" height="340">
</td>

**A**：蒸馏配置中的node需要设置成网络的输出节点。

1. 使用netron打开静态图模型model.pdmodel；
2. 修改QAT配置中node为最后一层卷积的输出名字。

<td>
<img src="https://github.com/PaddlePaddle/PaddleOCR/assets/34859558/b714040a-eec1-43df-af11-a233bd4cb59b" width="800" height="100">
</td>

<img width="800" alt="e589cdafd43796aed4c1b11c6828fefd" src="https://github.com/PaddlePaddle/PaddleOCR/assets/34859558/c75c0749-898b-4187-a9f9-363cfe92b1a4">


### 3. 量化蒸馏训练精度很低？
<img width="800" alt="2d916558811eb5f1bbb388025ddda21c" src="https://github.com/PaddlePaddle/PaddleOCR/assets/34859558/cc9bcc26-1568-4ab9-96f3-ff181486637c">


**A**：去除量化训练的输出结果，重新运行一次，这是由于网络训练到局部极值点导致。

### 4. TensorRT推理报错：TensorRT dynamic library not found.

<td>
<img src="https://user-images.githubusercontent.com/5997715/185016439-140e3c4a-002d-4c18-b0a8-d861a418d1e2.png" width="800" height="220">
</td>

**A**：参考[TensorRT安装说明](../../../docs/deployment/installtrt.md)，查看是否有版本不匹配或者路径没有配置。

### 5. ImportError: cannot import name 'MSRA' from 'paddle.fluid.initializer':

**A** 需要安装paddleslim 2.5，其适配了paddle2.5

### 6. ValueError: The axis is expected to be in range of [0,0) but got:

**A**: 需要安装paddleseg devleop版本，如果确定已经安装，建议使用`pip uninstall paddleseg`卸载后重新安装。

### 7. NotImplementedError：delete weight dequant op pass is not supported for per channel quantization

**A**：参考https://github.com/PaddlePaddle/Paddle/issues/56619，并参考[TensorRT安装说明](../../../docs/deployment/installtrt.md)安装TensorRT。

### 8. CPU推理精度严重下降

**A**：CPU推理精度下降通常是由于推理过程中量化的op设置问题导致的，请确保推理过程中量化的op和训练过程中量化的op一致，才能保证推理精度和训练精度对齐。以本文的`PP-Liteseg`为例进行说明：

量化训练配置文件是`configs/ppliteseg/ppliteseg_qat.yaml`，其中量化的op是`conv2d`和`depthwise_conv2d`，因此在推理过程中也需要量化这两个op，可以通过使用如下函数进行设置：
```python
# deploy/slim/act/test_seg.py:64
pred_cfg.enable_mkldnn_int8({
                    "conv2d", "depthwise_conv2d"
                })
```
而且最好只量化这两个op，如果增加其他op的量化，可能会导致精度下降。以下是一个简单的实验结果：

|        | 原模型fp32推理 | 原模型fp32+mkldnn加速 | 量化模型int8推理（量化conv2d,depthwise_conv2d） | 量化模型int8推理（量化conv2d,depthwise_conv2d,elementwise_mul） | 量化模型int8推理（量化conv2d,depthwise_conv2d,elementwise_mul,pool2d） |
|:------:|:---------:|:----------------:|:-------------------------------------:|:-----------------------------------------------------:|:------------------------------------------------------------:|
|  mIoU  |  0.7704   |      0.7704      |                0.7658                 |                        0.7657                         |                            0.7372                            |
| 耗时（ms） |  1216.8   |      1191.3      |                 434.5                 |                         439.6                         |                            505.8                             |

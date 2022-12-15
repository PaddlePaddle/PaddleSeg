简体中文 | [English](README.md)

# 遥感图像语义分割Benchmark

## 1 引言

语义分割被广泛地应用于遥感图像目标检测中，助力土地覆盖分类、灾难预测以及城市规划等等。本项目利用PaddleSeg进行遥感图像语义分割，主要贡献如下：

* **遥感图像语义分割Benchmark：** 向用户提供标准的数据处理、参数配置以及一套可比较的基线模型。
* **自监督学习：** 提供多个基于自监督学习方式预训练的模型，助力自监督在遥感领域的应用和研究。
* **一个从粗到精细化的分割模型：** 基于上述Benchmark提出一个优化小目标分割准确率的模型。[详情请见](./c2fnet/README.md)。


## 2 模型效果

### 2.1 基线模型

本项目提出的Benchmark在[iSAID](https://captain-whu.github.io/iSAID), [ISPRS Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx) 以及 [ISPRS Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)数据集上实现多个基线模型。结果如下：

#### 2.1.1 iSAID
| 模型 | 分辨率 | 主干网络 | 迭代次数 | mIoU(%) | 链接 |
| ----- | ---------- | ---------- | -----------------| ----------------- | ------- |
| DANet | 512x512 | ResNet50 | 80000 | 37.30 | [配置](./configs/danet/danet_resnet50_isaid_512x512_80k.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/danet_resnet50_isaid_512x512_80k/model.pdparams) |
| DANet | 512x512 | ResNet50_vd | 80000 | 64.56 | [配置](./configs/danet/danet_resnet50_vd_isaid_512x512_80k.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/danet_resnet50_vd_isaid_512x512_80k/model.pdparams) |
| DeeplabV3+ | 512x512 | ResNet50 | 80000 | 62.59 | [配置](./configs/deeplabv3p/deeplabv3p_resnet50_isaid_512x512_80k.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/deeplabv3p_resnet50_isaid_512x512_80k/model.pdparams) |
| DeeplabV3+ | 512x512 | ResNet50_vd | 80000 | 65.46 | [配置](./configs/deeplabv3p/deeplabv3p_resnet50_vd_isaid_512x512_80k.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/deeplabv3p_resnet50_vd_isaid_512x512_80k/model.pdparams) |
| FCN | 512x512 | HRNet_W18 | 80000 | 64.73 | [配置](./configs/fcn/fcn_hrnet_w18_isaid_512x512_80k.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/fcn_hrnet_w18_isaid_512x512_80k/model.pdparams) |
| FCN | 512x512 | ResNet50 | 80000 | 52.12 | [配置](./configs/fcn/fcn_resnet50_isaid_512x512_80k.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/fcn_resnet50_isaid_512x512_80k/model.pdparams) |
| HRNet | 512x512 | HRNet_W48 | 80000 | 67.31 | [配置](./configs/hrnet/hrnet_w48_isaid_512x512_80k.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/hrnet_w48_isaid_512x512_80k/model.pdparams) |
| PSPNet | 512x512 | ResNet50_vd | 80000 | 63.36 | [配置](./configs/pspnet/pspnet_resnet50_vd_isaid_512x512_80k.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/pspnet_resnet50_vd_isaid_512x512_80k/model.pdparams) |
| UperNet | 512x512 | ResNet50 | 80000 | 64.10 | [配置](./configs/upernet/upernet_resnet50_isaid_512x512_80k.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/upernet_resnet50_isaid_512x512_80k/model.pdparams) |

#### 2.1.2 ISPRS Potsdam

| 模型 | 分辨率 | 主干网络 | 迭代次数 | mIoU(%) | 链接 |
| ----- | ---------- | ---------- | -----------------| ----------------- | ------- |
| DeeplabV3+ | 512x512 | ResNet50_vd | 80000 | 77.93 | [配置](./configs/deeplabv3p/deeplabv3p_resnet50_vd_potsdam_512x512_80k.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/potsdam/deeplabv3p_resnet50_vd_potsdam_512x512_80k/model.pdparams) |
| FCN | 512x512 | HRNet_W18 | 80000 | 78.13 | [配置](./configs/fcn/fcn_hrnet_w18_potsdam_512x512_80k.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/potsdam/fcn_hrnet_w18_potsdam_512x512_80k/model.pdparams) |
| HRNet | 512x512 | HRNet_W48 | 80000 | 78.84 | [配置](./configs/hrnet/hrnet_w48_potsdam_512x512_80k.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/potsdam/hrnet_w48_potsdam_512x512_80k/model.pdparams) |
| PSPNet | 512x512 | ResNet50_vd | 80000 | 77.69 | [配置](./configs/pspnet/pspnet_resnet50_vd_potsdam_512x512_80k.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/potsdam/pspnet_resnet50_vd_potsdam_512x512_80k/model.pdparams) |
| UperNet | 512x512 | ResNet50 | 80000 | 77.59 | [配置](./configs/upernet/upernet_resnet50_potsdam_512x512_80k.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/potsdam/upernet_resnet50_potsdam_512x512_80k/model.pdparams) |



#### 2.1.3 ISPRS Vaihingen

| 模型 | 分辨率 | 主干网络 | 迭代次数 | mIoU(%) | 链接 |
| ----- | ---------- | ---------- | -----------------| ----------------- | ------- |
| DeeplabV3+ | 512x512 | ResNet50_vd | 80000 | 74.08 | [配置](./configs/deeplabv3p/deeplabv3p_resnet50_vd_vaihingen_512x512_80k.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/vaihingen/deeplabv3p_resnet50_vd_vaihingen_512x512_80k/model.pdparams) |
| FCN | 512x512 | HRNet_W18 | 80000 | 73.25 | [配置](./configs/fcn/fcn_hrnet_w18_vaihingen_512x512_80k.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/vaihingen/fcn_hrnet_w18_vaihingen_512x512_80k/model.pdparams)|
| HRNet | 512x512 | HRNet_W48 | 80000 | 74.98 | [配置](./configs/hrnet/hrnet_w48_vaihingen_512x512_80k.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/vaihingen/hrnet_w48_vaihingen_512x512_80k/model.pdparams) |
| UperNet | 512x512 | ResNet50_vd | 80000 | 74.31 | [配置](./configs/upernet/upernet_resnet50_vd_vaihingen_512x512_80k.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/vaihingen/upernet_resnet50_vd_vaihingen_512x512_80k/model.pdparams)|}

### 2.2 自监督预训练模型

我们基于[PASSL](https://github.com/PaddlePaddle/PASSL)研究了自监督学习在遥感图像的泛化能力。本项目向用户提供一些有价值的实验结果和多个自监督预训练模型，便于用户的进一步研究。

#### 2.2.1 基于ImageNet的自监督预训练模型

| 数据集 | 分割器 | 自监督方法 | 主干网络 | mIoU(%) | 链接 |
| ----- | ---------- | ---------- | -----------------| ----------------- | ------- |
| iSAID | DeeplabV3+ | DenseCL | ResNet50 | 56.94 | [配置](./configs/ssl/deeplabv3p_densecl_imgnet_resnet50_isaid_512x512.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/deeplabv3p_densecl_imgnet_resnet50_isaid_512x512/model.pdparams) |
| iSAID | DeeplabV3+ | MoCoBYOL | ResNet50 | 57.96 | [配置](./configs/ssl/deeplabv3p_mocobyol_imgnet_resnet50_isaid_512x512.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/deeplabv3p_mocobyol_imgnet_resnet50_isaid_512x512/model.pdparams) |
| iSAID | DeeplabV3+ | PixelPro | ResNet50 | 62.22 | [配置](./configs/ssl/deeplabv3p_pixpro_imgnet_resnet50_isaid_512x512.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/deeplabv3p_pixpro_imgnet_resnet50_isaid_512x512/model.pdparams) |
| iSAID | FCN | PixelPro | ResNet50 | 51.30 | [配置](./configs/ssl/fcn_pixpro_imgnet_resnet50_isaid_512x512.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/fcn_pixpro_imgnet_resnet50_isaid_512x512/model.pdparams) |
| iSAID | OCRNet | PixelPro | ResNet50 | 41.95 | [配置](./configs/ssl/ocrnet_pixpro_imgnet_resnet50_isaid_512x512.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/ocrnet_pixpro_imgnet_resnet50_isaid_512x512/model.pdparams) |
| iSAID | PSPNet | PixelPro | ResNet50 | 50.23 | [配置](./configs/ssl/pspnet_pixpro_imgnet_resnet50_isaid_512x512.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/pspnet_pixpro_imgnet_resnet50_isaid_512x512/model.pdparams) |
| iSAID | UperNet | DenseCL | ResNet50 | 54.22 | [配置](./configs/ssl/upernet_densecl_imgnet_resnet50_isaid_512x512.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/upernet_densecl_imgnet_resnet50_isaid_512x512/model.pdparams) |
| iSAID | UperNet | MoCoBYOL | ResNet50 | 64.36 | [配置](./configs/ssl/upernet_mocobyol_imgnet_resnet50_isaid_512x512.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/upernet_mocobyol_imgnet_resnet50_isaid_512x512/model.pdparams) |
| iSAID | UperNet | PixelPro | ResNet50 | 64.36 | [配置](./configs/ssl/upernet_pixpro_imgnet_resnet50_isaid_512x512.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/upernet_pixpro_imgnet_resnet50_isaid_512x512/model.pdparams) |
| iSAID | UperNet | SimSiam | ResNet50 | 50.70 | [配置](./configs/ssl/upernet_simsiam_imgnet_resnet50_isaid_512x512.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/upernet_simsiam_imgnet_resnet50_isaid_512x512/model.pdparams) |
| iSAID | UperNet | SwAV | ResNet50 | 63.42 | [配置](./configs/ssl/upernet_swav_imgnet_resnet50_isaid_512x512.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/upernet_swav_imgnet_resnet50_isaid_512x512/model.pdparams) |
| Potsdam | UperNet | PixelPro | ResNet50 | 77.40 | [配置](./configs/ssl/upernet_pixpro_imgnet_resnet50_potsdam_512x512.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/upernet_pixpro_imgnet_resnet50_potsdam_512x512/model.pdparams) |

*注意： 所有上述的自监督预训练主干网络全部来自[PASSL](https://github.com/PaddlePaddle/PASSL)。*

#### 2.2.2 基于遥感图像的自监督预训练模型

我们在[Million-AID](https://paperswithcode.com/dataset/million-aid) 和 [DOTA2.0](https://captain-whu.github.io/DOTA/dataset.html)两个遥感图像数据集上应用自监督学习方法。为了获得充足的遥感数据，我们将两个遥感数据集内不同分辨率的图像剪裁至512x512。剪裁后的Million-AID数据集包含 **2.5M** 张遥感图像块；DOTA2.0数据集包含 **1.7M** 张遥感图像块。

| 数据集 | 分割器 | 自监督方法 | 主干网络 | mIoU(%) | 链接 |
| ----- | ---------- | ---------- | -----------------| ----------------- | ------- |
| iSAID | FCN | PixelPro DOTA | ResNet50 | 41.26 | [配置](./configs/ssl/fcn_pixpro_dota_resnet50_isaid_512x512.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/fcn_pixpro_dota_resnet50_isaid_512x512/model.pdparams)|
| iSAID | UperNet | DenseCL Million-AID | ResNet50 | 61.67 | [配置](./configs/ssl/upernet_densecl_millionaid_resnet50_isaid_512x512.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/upernet_densecl_millionaid_resnet50_isaid_512x512/model.pdparams) |
| iSAID | UperNet | MoCoV2 Million-AID | ResNet50 | 55.62 | [配置](./configs/ssl/upernet_mocov2_millionaid_resnet50_isaid_512x512.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/upernet_mocov2_millionaid_resnet50_isaid_512x512/model.pdparams) |
| iSAID | UperNet | PixelPro DOTA | ResNet50 | 59.50 |[配置](./configs/ssl/upernet_pixpro_dota_resnet50_isaid_512x512.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/upernet_pixpro_dota_resnet50_isaid_512x512/model.pdparams) |
| iSAID | UperNet | PixelPro Million-AID | ResNet50 | 58.24 | [配置](./configs/ssl/upernet_pixpro_millionaid_resnet50_isaid_512x512.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/upernet_pixpro_millionaid_resnet50_isaid_512x512/model.pdparams) |
| Potsdam | UperNet | PixelPro Million-AID | ResNet50 | 75.68 | [配置](./configs/ssl/upernet_pixpro_millionaid_resnet50_potsdam_512x512.yml) \| [模型](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/upernet_pixpro_millionaid_resnet50_potsdam_512x512/model.pdparams) |

## 3 安装

### 3.1 环境依赖

* Python: 3.7+  
* PaddlePaddle: 2.3.2
* PaddleSeg: 2.6


### 3.2 安装
a. 创建一个Anaconda虚拟环境并激活它。
```shell
conda create -n rsseg python=3.7
conda activate rsseg
```

b. 安装[PadddlePaddle](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/install/pip/linux-pip_en.html)  (版本需 >= 2.3)。

c. 下载PaddleSeg库。
```shell
git clone https://github.com/PaddlePaddle/PaddleSeg
```

d. 安装[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/docs/install.md)以及所需环境依赖。

f. 进入Benchmark目录。

```shell
cd PaddleSeg/contrib/RSSegBenchmark
```


## 4 数据集准备

a. 下载数据集。

+ [iSAID](https://captain-whu.github.io/iSAID)
+ [ISPR Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)
+ [ISPRS Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)

b. 预处理数据集。

处理 iSAID。

```python
python data/prepare_isaid.py {PATH OF ISAID}
```

处理 ISPRS Potsdam。

```python
python data/prepare_potsdam.py {PATH OF POTSDAM}
```

处理 ISPRS Vaihingen。

```python
python data/prepare_vaihingen.py {PATH OF VAIHINGEN}
```

*注意：`train.txt`、`val.txt`、`label.txt` 文件的生成需要参考[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/docs/data/marker/marker_cn.md)。*


## 5 模型训练和验证

### 5.1 训练
单个GPU训练。

```shell
export CUDA_VISIBLE_DEVICES=0
python train.py \
       --config configs/{YOUR CONFIG FILE} \
       --do_eval \
       --save_interval 8000 \
       --save_dir {OUTPUT PATH}
```
多GPU训练。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch train.py \
      --config configs/{YOUR CONFIG FILE} \
      --do_eval \
      --save_interval 8000 \
      --save_dir {OUTPUT PATH}
```

*注意：更多训练的设置和细节请到[PaddleSeg文档](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/docs/train/train.md)内查看。*

### 5.2 验证

得到最优秀模型的评价指标。

```shell
python val.py \
      --config configs/{YOUR CONFIG FILE} \
      --model_path {YOUR BEST MODEL PATH}
```
*注意：更多细节参考[PaddleSeg文档](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/docs/evaluation/evaluate.md)。*

### 5.3 预测

预测以及保存最优模型的分割结果。

```shell
python predict.py \
       --config configs/{YOUR CONFIG FILE} \
       --model_path {YOUR BEST MODEL PATH}
       --image_path {IMAGE PATH}\
       --save_dir {OUTPUT DIR}
```
*注意：更多细节参考[PaddleSeg文档](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/docs/predict/predict.md)。*


## 联系人

wangqingzhong@baidu.com

silin.chen@cumt.edu.cn

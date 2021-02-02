
PaddlePaddle implementation of the paper [Hierarchical Multi-Scale Attention for Semantic Segmentation](https://arxiv.org/abs/2005.10821).<br>
基于Hierarchical Multi-Scale Attention for Semantic Segmentation，我们做了一些优化

## Installation

#### step 1. Install PaddlePaddle

System Requirements:
* PaddlePaddle >= 2.0.0rc
* Python >= 3.6+

Highly recommend you install the GPU version of PaddlePaddle, due to large overhead of segmentation models, otherwise it could be out of memory while running the models. For more detailed installation tutorials, please refer to the official website of [PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-beta/install/index_cn.html)。


#### step 2. Install PaddleSeg
Support to construct a customized segmentation framework with *API Calling* method for flexible development.

```shell
pip install paddleseg
```

## Data Preparation
Firstly please download 3 files from [Cityscapes dataset](https://www.cityscapes-dataset.com/downloads/)
- leftImg8bit_trainvaltest.zip
- gtFine_trainvaltest.zip
- leftImg8bit_trainextra.zip

Then download Autolabelled-Data from [google drive](https://drive.google.com/file/d/1DtPo-WP-hjaOwsbj6ZxTtOo_7R_4TKRG/view?usp=sharing)
- refinement_final_v0.zip

You need to unzip these files, and organize data following the below structure.

    cityscapes
    |
    |--leftImg8bit
    |  |--train
    |  |--val
    |  |--test
    |
    |--gtFine
    |  |--train
    |  |--val
    |  |--test
    |
    |--leftImg8bit_trainextra
    |  |--leftImg8bit
    |     |--train_extra
    |        |--augsburg
    |        |--bayreuth
    |        |--...
    |
    |--convert_autolabelled
    |  |--augsburg
    |  |--bayreuth
    |  |--...

## Download Pretrained Weights

```shell
mkdir -p pretrain && cd pretrain
wget https://bj.bcebos.com/paddleseg/dygraph/cityscapes/ocrnet_hrnetw48_mapillary/pretrained.pdparams
cd ..
```

Pretrained weights were obtained by pretraining on the Mapillary dataset from OCRNet (backbone is HRNet w48).

## Run Inference on Cityscapes

### Download Trained Model
```shell
mkdir -p saved_model && cd saved_model
wget https://bj.bcebos.com/paddleseg/dygraph/cityscapes/mscale_ocr_hrnetw48_cityscapes_autolabel_mapillary/model.pdparams
cd ..
```
### Evaluate
```


## Prediction effect
![demo](../../docs/images/cityscapes.gif)

2020-12-24 06:40:53 [INFO]  [EVAL] #Images=500 mIoU=0.8700 Acc=0.9756 Kappa=0.9683  
2020-12-24 06:40:53 [INFO]  [EVAL] Class IoU:  
[0.9902 0.9165 0.951  0.7518 0.7957 0.7701 0.8016 0.8758 0.9418 0.7497
 0.9611 0.8795 0.7442 0.9688 0.9207 0.957  0.9162 0.8    0.8383]
2020-12-24 06:40:53 [INFO]  [EVAL] Class Acc:  
[0.9958 0.9565 0.9751 0.8854 0.875  0.8733 0.8787 0.9354 0.9675 0.8899
 0.9768 0.9302 0.8448 0.9833 0.9609 0.98   0.9714 0.8966 0.8985]


 2020-12-24 06:29:40 [INFO]  [EVAL] #Images=500 mIoU=0.8699 Acc=0.9756 Kappa=0.9684
2020-12-24 06:29:40 [INFO]  [EVAL] Class IoU:
[0.9902 0.9168 0.9511 0.7515 0.7966 0.7708 0.8016 0.8756 0.9417 0.7498
 0.961  0.8796 0.7437 0.9689 0.9207 0.9578 0.917  0.7963 0.8383]
2020-12-24 06:29:40 [INFO]  [EVAL] Class Acc:
[0.9958 0.9564 0.975  0.8858 0.8751 0.8741 0.8788 0.9358 0.9677 0.8899
 0.9766 0.9294 0.8445 0.9832 0.9617 0.9808 0.974  0.8943 0.8992]

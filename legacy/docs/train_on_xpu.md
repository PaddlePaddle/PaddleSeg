# 图像分割昆仑模型介绍(持续更新中)

## 前言

* 本文档介绍了目前昆仑支持的图像分割模型以及如何在昆仑设备上训练这些模型

## 昆仑训练

### DeeplabV3
* 数据准备(在legacy目录下)：

```shell
python dataset/download_optic.py
```

* 预训练模型准备(在legacy目录下)：

```shell
python pretrained_model/download_model.py deeplabv3p_xception65_bn_coco
```

* 执行训练(在legacy目录下)：

```shell
python pdseg/train.py --cfg configs/deeplabv3p_xception65_optic_kunlun.yaml --use_mpio --use_xpu --log_steps 1 --do_eval
```

### Unet
* 数据准备(在legacy目录下)：

```shell
python dataset/download_optic.py
```

* 预训练模型准备(在legacy目录下)：

```shell
python pretrained_model/download_model.py unet_bn_coco
```

* 执行训练(在legacy目录下)：

因为昆仑1的内存不够，在用昆仑1训练的时候，需要把./configs/unet_optic.yaml 里面的 BATCH_SIZE
修改为 1

```shell
# 指定xpu的卡号 （以0号卡为例）
export FLAGS_selected_xpus=0
# 执行xpu产品名称 这里指定昆仑1
export XPUSIM_DEVICE_MODEL=KUNLUN1
# 训练
python pdseg/train.py --use_xpu --cfg configs/unet_optic.yaml --use_mpio --log_steps 1 --do_eval
```

### FCN
* 数据准备(在legacy目录下)：

```shell
python dataset/download_optic.py
```

* 预训练模型准备(在legacy目录下)：

```shell
python pretrained_model/download_model.py hrnet_w18_bn_cityscapes
```

* 执行训练(在legacy目录下)：

因为昆仑1的内存不够，在用昆仑1训练的时候，需要把./configs/fcn.yaml 里面的 BATCH_SIZE
修改为 1

```shell
# 指定xpu的卡号 （以0号卡为例）
export FLAGS_selected_xpus=0
# 执行xpu产品名称 这里指定昆仑1
export XPUSIM_DEVICE_MODEL=KUNLUN1
# 训练
export PYTHONPATH=`pwd`
python3 pdseg/train.py --cfg configs/fcn.yaml --use_mpio --log_steps 1 --do_eval
```

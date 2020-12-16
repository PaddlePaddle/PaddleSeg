# 图像分割昆仑模型介绍(持续更新中)

## 前言

* 本文档介绍了目前昆仑支持的图像分割模型以及如何在昆仑设备上训练这些模型

## 昆仑训练

### DeeplabV3
* 数据准备：

```cd PaddleSeg/legacy;python pretrained_model/download_model.py deeplabv3p_xception65_bn_coco```

* 预训练模型准备：

```cd PaddleSeg/legacy;python3.7 dataset/download_optic.py```


* 执行训练：

```cd PaddleSeg/legacy;python3.7 pdseg/train.py --cfg configs/deeplabv3p_xception65_optic_kunlun.yaml --use_mpio --use_xpu --log_steps 1 --do_eval```


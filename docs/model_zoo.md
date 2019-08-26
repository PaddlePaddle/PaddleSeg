# PaddleSeg 预训练模型

PaddleSeg对所有内置的分割模型都提供了公开数据集的下的预训练模型，通过加载预训练模型后训练可以在自定义数据集中得到更稳定地效果。

## ImageNet预训练模型 

所有Imagenet预训练模型来自于PaddlePaddle图像分类库，想获取更多细节请点击[这里](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification))

| 模型 | 数据集合 | Depth multiplier | 模型加载config设置 | 下载地址 | Accuray Top1/5 Error|
|---|---|---|---|---|---|
| MobieNetV2_1.0x  | ImageNet | 1.0x  | MODEL.MODEL_NAME: deeplabv3p <br> MODEL.DEEPLAB.BACKBONE: mobilenet <br> MODEL.DEEPLAB.DEPTH_MULTIPLIER: 1.0 <br> MODEL.DEFAULT_NORM_TYPE: bn| [MobileNetV2_1.0x] (https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_pretrained.tar) | 72.15%/90.65% |
| MobieNetV2_0.25x | ImageNet | 0.25x | MODEL.MODEL_NAME: deeplabv3p <br> MODEL.DEEPLAB.BACKBONE: mobilenet <br> MODEL.DEEPLAB.DEPTH_MULTIPLIER: 0.25 <br> MODEL.DEFAULT_NORM_TYPE: bn |[MobileNetV2_0.25x] (https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x0_25_pretrained.tar) | 53.21%/76.52% |
| MobieNetV2_0.5x  | ImageNet | 0.5x  | MODEL.MODEL_NAME: deeplabv3p <br> MODEL.DEEPLAB.BACKBONE: mobilenet <br> MODEL.DEEPLAB.DEPTH_MULTIPLIER: 0.5 <br> MODEL.DEFAULT_NORM_TYPE: bn | [MobileNetV2_0.5x] (https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x0_5_pretrained.tar) | 65.03%/85.72% |
| MobieNetV2_1.5x  | ImageNet | 1.5x  | MODEL.MODEL_NAME: deeplabv3p <br> MODEL.DEEPLAB.BACKBONE: mobilenet <br> MODEL.DEEPLAB.DEPTH_MULTIPLIER: 1.5 <br> MODEL.DEFAULT_NORM_TYPE: bn| [MobileNetV2_1.5x] (https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x1_5_pretrained.tar) | 74.12%/91.67% |
| MobieNetV2_2.0x  | ImageNet | 2.0x  | MODEL.MODEL_NAME: deeplabv3p <br> MODEL.DEEPLAB.BACKBONE: mobilenet <br> MODEL.DEEPLAB.DEPTH_MULTIPLIER: 2.0 <br> MODEL.DEFAULT_NORM_TYPE: bn | [MobileNetV2_2.0x] (https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x2_0_pretrained.tar) | 75.23%/92.58% |

用户可以结合实际场景的精度和预测性能要求，选取不同`Depth multiplier`参数的MobileNet模型。

| 模型 | 数据集合 | 模型加载config设置 | 下载地址 | Accuray Top1/5 Error |
|---|---|---|---|---|
| Xception41 | ImageNet | MODEL.MODEL_NAME: deeplabv3p <br> MODEL.DEEPLAB.BACKBONE: xception_41 <br> MODEL.DEFAULT_NORM_TYPE: bn| [Xception41_pretrained.tgz] (https://paddleseg.bj.bcebos.com/models/Xception41_pretrained.tgz) | 79.5%/94.38% |
| Xception65 | ImageNet | MODEL.MODEL_NAME: deeplabv3p <br> MODEL.DEEPLAB.BACKBONE: xception_65 <br> MODEL.DEFAULT_NORM_TYPE: bn| [Xception65_pretrained.tgz] (https://paddleseg.bj.bcebos.com/models/Xception65_pretrained.tgz) | 80.32%/94.47% |
| Xception71 | ImageNet | MODEL.MODEL_NAME: deeplabv3p <br> MODEL.DEEPLAB.BACKBONE: xception_71 <br> MODEL.DEFAULT_NORM_TYPE: bn| coming soon | -- |

## COCO预训练模型 

train数据集为coco instance分割数据集合转换成的语义分割数据集合

| 模型 | 数据集合 | 模型加载config设置 | 下载地址 |Output Strid|multi-scale test| mIoU |
|---|---|---|---|---|---|---|
| DeepLabv3+/MobileNetv2/bn | COCO | MODEL.MODEL_NAME: deeplabv3p <br> MODEL.DEEPLAB.BACKBONE: mobilenet <br> MODEL.DEEPLAB.DEPTH_MULTIPLIER: 1.0 <br> MODEL.DEFAULT_NORM_TYPE: bn|[deeplabv3plus_coco_bn_init.tgz](https://bj.bcebos.com/v1/paddleseg/deeplabv3plus_coco_bn_init.tgz) | 16 | --| -- |
| DeeplabV3+/Xception65/bn | COCO | MODEL.MODEL_NAME: deeplabv3p <br> MODEL.DEEPLAB.BACKBONE: xception_65 <br> MODEL.DEFAULT_NORM_TYPE: bn | [xception65_coco.tgz](https://paddleseg.bj.bcebos.com/models/xception65_coco.tgz)| 16 | -- | -- |
| UNet/bn | COCO | MODEL.MODEL_NEME: unet  <br> MODEL.DEFAULT_NORM_TYPE: bn | [unet](https://paddleseg.bj.bcebos.com/models/unet_coco_v2.tgz) | 16 | -- | -- |

## Cityscapes预训练模型 

train数据集合为Cityscapes 训练集合，测试为Cityscapes的验证集合

| 模型 | 数据集合 | 模型加载config设置 | 下载地址 |Output Stride| mutli-scale test| mIoU on val|
|---|---|---|---|---|---|---|---|
| DeepLabv3+/MobileNetv2/bn | Cityscapes |MODEL.MODEL_NAME: deeplabv3p <br> MODEL.DEEPLAB.BACKBONE: mobilenet <br> MODEL.DEEPLAB.DEPTH_MULTIPLIER: 1.0 <br> MODEL.DEEPLAB.ENCODER_WITH_ASPP: False <br> MODEL.DEEPLAB.ENABLE_DECODER: False <br> MODEL.DEFAULT_NORM_TYPE: bn|[mobilenet_cityscapes.tgz] (https://paddleseg.bj.bcebos.com/models/mobilenet_cityscapes.tgz) |16|false| 0.698|
| DeepLabv3+/Xception65/gn  | Cityscapes |MODEL.MODEL_NAME: deeplabv3p <br> MODEL.DEEPLAB.BACKBONE: xception_65 <br> MODEL.DEFAULT_NORM_TYPE: gn | [deeplabv3p_xception65_cityscapes.tgz](https://paddleseg.bj.bcebos.com/models/deeplabv3p_xception65_cityscapes.tgz) |16|false| 0.7804 |
| DeepLabv3+/Xception65/bn | Cityscapes | MODEL.MODEL_NAME: deeplabv3p <br>  MODEL.DEEPLAB.BACKBONE: xception_65 <br> MODEL.DEFAULT_NORM_TYPE: bn| [Xception65_deeplab_cityscapes.tgz](https://paddleseg.bj.bcebos.com/models/Xception65_deeplab_cityscapes.tgz) | 16 | false | 0.7715 |
| ICNet/bn | Cityscapes | MODEL.MODEL_NAME: icnet <br> MODEL.DEFAULT_NORM_TYPE: bn |  [icnet_cityscapes.tgz](https://paddleseg.bj.bcebos.com/models/icnet_cityscapes.tgz) |16|false| 0.6854 |

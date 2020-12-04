# Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation

## Reference

> Chen, Liang-Chieh, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. "Encoder-decoder with atrous separable convolution for semantic image segmentation." In Proceedings of the European conference on computer vision (ECCV), pp. 801-818. 2018.

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|DeepLabV3P|ResNet50_OS8|1024x512|80000|80.36%|80.57%|80.81%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=860bd0049ba5495d629a96d5aaf1bf75)|
|DeepLabV3P|ResNet101_OS8|1024x512|80000|81.10%|81.38%|81.24%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/deeplabv3p_resnet101_os8_cityscapes_1024x512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/deeplabv3p_resnet101_os8_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=8b11e75b8977a0fd74180145350c27de)|
|DeepLabV3P|ResNet101_OS8|769x769|80000|81.53%|81.88%|82.12%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/deeplabv3p_resnet101_os8_cityscapes_769x769_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/deeplabv3p_resnet101_os8_cityscapes_769x769_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=420039406361cbc3cf7ec14c1084d886)|

### Pascal VOC 2012 + Aug

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|DeepLabV3P|ResNet50_OS8|512x512|40000|80.66%|81.33%|81.49%|[model](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/deeplabv3p_resnet50_os8_voc12aug_512x512_40k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/deeplabv3p_resnet50_os8_voc12aug_512x512_40k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=a2891ac5fb866b3ea8c38289e5b1d686)|
|DeepLabV3P|ResNet101_OS8|512x512|40000|80.60%|80.77%|80.75%|[model](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/deeplabv3p_resnet101_os8_voc12aug_512x512_40k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/deeplabv3p_resnet101_os8_voc12aug_512x512_40k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=304048e5c2b57949f56b75b88ccb5645)|

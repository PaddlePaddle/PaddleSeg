# Rethinking Atrous Convolution for Semantic Image Segmentation

## Reference

> Chen, Liang-Chieh, George Papandreou, Florian Schroff, and Hartwig Adam. "Rethinking Atrous Convolution for Semantic Image Segmentation." arXiv preprint arXiv:1706.05587 (2017).

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (multi-scale) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|DeepLabV3|ResNet50_OS8|1024x512|160000|79.60%|-|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/deeplabv3_resnet50_os8_cityscapes_1024x512_160k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/deeplabv3_resnet50_os8_cityscapes_1024x512_160k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=9dfd191d01883cbd0e5a910def16a758)|
|DeepLabV3|ResNet101_OS8|1024x512|160000|80.05%|-|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/deeplabv3_resnet101_os8_cityscapes_1024x512_160k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/deeplabv3_resnet101_os8_cityscapes_1024x512_160k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=67192fd2fa1f2428afc4f5cab19ecb07)|

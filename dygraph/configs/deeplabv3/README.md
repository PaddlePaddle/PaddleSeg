# Rethinking Atrous Convolution for Semantic Image Segmentation

## Reference

> Chen, Liang-Chieh, George Papandreou, Florian Schroff, and Hartwig Adam. "Rethinking Atrous Convolution for Semantic Image Segmentation." arXiv preprint arXiv:1706.05587 (2017).

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|DeepLabV3|ResNet50_OS8|1024x512|80000|79.90%|80.22%|80.47%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/deeplabv3_resnet50_os8_cityscapes_1024x512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/deeplabv3_resnet50_os8_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=7e30d1cb34cd94400e1e1266538dfb6c)|
|DeepLabV3|ResNet101_OS8|1024x512|80000|80.85%|81.09%|81.54%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/deeplabv3_resnet101_os8_cityscapes_1024x512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/deeplabv3_resnet101_os8_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=1ff25b7f3c5e88a051b9dd273625f942)|

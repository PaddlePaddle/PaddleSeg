# Object-Contextual Representations for Semantic Segmentation

## Reference

> Yuan Y, Chen X, Wang J. Object-contextual representations for semantic segmentation[J]. arXiv preprint arXiv:1909.11065, 2019.

## Performance

### CityScapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|OCRNet|HRNet_w18|1024x512|160000|80.67%|81.21%|81.30|[model](https://paddleseg.bj.bcebos.com/dygraph/ocrnet_hrnetw18_cityscapes_1024x512_160k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/ocrnet_hrnetw18_cityscapes_1024x512_160k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=901a5d0a78b71ca56f06002f05547837)|
|OCRNet|HRNet_w48|1024x512|160000|82.15%|82.59%|82.85|[model](https://paddleseg.bj.bcebos.com/dygraph/ocrnet_hrnetw48_cityscapes_1024x512_160k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/ocrnet_hrnetw48_cityscapes_1024x512_160k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=176bf6ca4d89957ffe62ac7c30fcd039) |

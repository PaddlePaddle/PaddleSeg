# Object-Contextual Representations for Semantic Segmentation

## Reference

> Yuan, Yuhui, Xilin Chen, and Jingdong Wang. "Object-contextual representations for semantic segmentation." arXiv preprint arXiv:1909.11065 (2019).

## Performance

### CityScapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|OCRNet|HRNet_w18|1024x512|160000|80.67%|81.21%|81.30%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/ocrnet_hrnetw18_cityscapes_1024x512_160k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/ocrnet_hrnetw18_cityscapes_1024x512_160k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=901a5d0a78b71ca56f06002f05547837)|
|OCRNet|HRNet_w48|1024x512|160000|82.15%|82.59%|82.85%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/ocrnet_hrnetw48_cityscapes_1024x512_160k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/ocrnet_hrnetw48_cityscapes_1024x512_160k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=176bf6ca4d89957ffe62ac7c30fcd039) |

### Pascal VOC 2012 + Aug

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|OCRNet|HRNet_w18|512x512|40000|75.76%|76.39%|77.95%|[model](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/ocrnet_hrnetw18_voc12aug_512x512_40k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/ocrnet_hrnetw18_voc12aug_512x512_40k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=74707b83bc14b7d236146ac4ceaf6c9c)|
|OCRNet|HRNet_w48|512x512|40000|79.98%|80.47%|81.02%|[model](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/ocrnet_hrnetw48_voc12aug_512x512_40k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/ocrnet_hrnetw48_voc12aug_512x512_40k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=8f695743c799f8966a72973f3259fad4) |

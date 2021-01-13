# Improving Semantic Segmentation via Decoupled Body and Edge Supervision

## Reference

> Li, Xiangtai, Xia Li, Li Zhang, Guangliang Cheng, Jianping Shi, Zhouchen Lin, Shaohua Tan, and Yunhai Tong. "Improving semantic segmentation via decoupled body and edge supervision." arXiv preprint arXiv:2007.10035 (2020).

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|DecoupledSegNet|ResNet50_OS8|1024x512|80000|80.86%|81.34%|81.49%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/decoupledsegnet_resnet50_os8_cityscapes_1024x512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/decoupledsegnet_resnet50_os8_cityscapes_1024x512_80k/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=3c5cba5e6f89b33dc75b43c62026dc12)|
|DecoupledSegNet|ResNet50_OS8|832x832|80000|81.26%|81.56%|81.80%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/decoupledsegnet_resnet50_os8_cityscapes_832x832_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/decoupledsegnet_resnet50_os8_cityscapes_832x832_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=e3e8f9044d96a57f7337f5928f2c265f)|

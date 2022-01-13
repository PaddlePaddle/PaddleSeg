# Semantic Flow for Fast and Accurate Scene Parsing

## Reference

> Xiangtai Li, Ansheng You, Zhen Zhu, Houlong Zhao, Maoke Yang, Kuiyuan Yang, Shaohua Tan, Yunhai Tong:
Semantic Flow for Fast and Accurate Scene Parsing. ECCV (1) 2020: 775-793 .

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|SFNet|ResNet18_OS8|1024x1024|80000|78.72%|79.11%|79.28%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/sfnet_resnet18_os8_cityscapes_1024x1024_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/sfnet_resnet18_os8_cityscapes_1024x1024_80k/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=0d790ad96282048b136342fcebb08d14)|
|SFNet|ResNet50_OS8|1024x1024|80000|81.49%|81.63%|81.85%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/sfnet_resnet50_os8_cityscapes_1024x1024_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/sfnet_resnet50_os8_cityscapes_1024x1024_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=d458349ec63ea8ccd6fae84afa8ea981)|

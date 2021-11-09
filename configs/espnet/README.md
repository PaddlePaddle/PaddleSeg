# ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network

## Reference

> Mehta, Sachin, Mohammad Rastegari, Linda Shapiro, and Hannaneh Hajishirzi. "Espnetv2: A light-weight, power efficient, and general purpose convolutional neural network." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9190-9200. 2019.

## Performance

### CityScapes

| Model | Backbone | Resolution | Training Iters | mIoU | Links |
|:---:|:---:|:---:|:---:|:---:|:---:|
|ESPNetV2|-|1024x512|120000|70.88%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/espnet_cityscapes_1024x512_120k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/espnet_cityscapes_1024x512_120k/train.log) \|[vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=c717bd8c2b5a083de759492158c14ffd)


# 其他说明
1、paddlepaddle==2.1.2 版本交叉熵损失函数有 bug，请在 develop 版本运行；  
2、paddleseg=develop 在 paddlepaddle==2.1.2 交叉熵损失函数传入 weight 时有 bug，请更换为 paddleseg-release2.2 下的交叉熵损失。  

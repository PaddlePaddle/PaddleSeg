# PIDNet: A Real-time Semantic Segmentation Network Inspired from PID Controller

## Reference

> Xu, Jiacong, Zixiang Xiong, and Shankar P. Bhattacharyya. "PIDNet: A Real-Time Semantic Segmentation Network Inspired by PID Controllers." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|PIDNet|PIDNet-Small |1024x1024|120000|78.48%|79.02%|79.68%|[model](https://paddleseg.bj.bcebos.com/dygraph/pidnet/pidnet_small_cityscapes_1024x1024_120k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/pidnet/pidnet_small_cityscapes_1024x1024_120k/pidnet_small.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=57dda9c34cd06a4b2996118df03583c9)|
|PIDNet|PIDNet_Medium|1024x1024|120000| | | |
|PIDNet|PIDNet-Large |1024x1024|120000| | | |


#### official weight

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|PIDNet|PIDNet-Small |1024x1024|120000|78.74%|79.53%|80.28%|[model](https://paddleseg.bj.bcebos.com/dygraph/pidnet/pidnet_small_2xb6-120k_1024x1024-cityscapes.pdparams)|
|PIDNet|PIDNet_Medium|1024x1024|120000|80.22%|81.07%|81.50%|[model](https://paddleseg.bj.bcebos.com/dygraph/pidnet/pidnet_medium_2xb6-120k_1024x1024-cityscapes.pdparams)|
|PIDNet|PIDNet-Large |1024x1024|120000|80.89%|81.41%|81.92%|[model](https://paddleseg.bj.bcebos.com/dygraph/pidnet/pidnet_large_2xb6-120k_1024x1024-cityscapes.pdparams)|


#### NOTE:

The `weight` in the training parameters is dataset dependent, if you are using another dataset, modify the weight please.

``` yaml
# follow the OCNet, compute the weights by 1/log(pixel_count)
# see https://github.com/openseg-group/OCNet.pytorch/issues/14
weight: &weight [0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
                 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                 1.0865, 1.0955, 1.0865, 1.1529, 1.0507]
```

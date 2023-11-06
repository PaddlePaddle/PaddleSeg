# PIDNet: A Real-time Semantic Segmentation Network Inspired from PID Controller

## Reference

> Xu, Jiacong, Zixiang Xiong, and Shankar P. Bhattacharyya. "PIDNet: A Real-Time Semantic Segmentation Network Inspired by PID Controllers." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|
|PIDNet|PIDNet-Small |1024x1024|120000|78.74%|80.87%|[model](https://paddleseg.bj.bcebos.com/dygraph/pidnet/pidnet_small_cityscapes_1024x1024_120k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/pidnet/pidnet_small_cityscapes_1024x1024_120k/pidnet_small.log)\| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=57dda9c34cd06a4b2996118df03583c9)|
|PIDNet|PIDNet_Medium|1024x1024|120000|80.22%|82.05%|[model](https://paddleseg.bj.bcebos.com/dygraph/pidnet/pidnet_medium_2xb6-120k_1024x1024-cityscapes.pdparams)|
|PIDNet|PIDNet-Large |1024x1024|120000|80.89%|82.37%|[model](https://paddleseg.bj.bcebos.com/dygraph/pidnet/pidnet_large_2xb6-120k_1024x1024-cityscapes.pdparams)|

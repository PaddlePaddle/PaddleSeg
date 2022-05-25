# MobileSeg

These semantic segmentation models are designed for mobile and edge devices.

## Reference

> Sandler, Mark, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. "Mobilenetv2: Inverted residuals and linear bottlenecks." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4510-4520. 2018.
> Howard, Andrew, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang et al. "Searching for mobilenetv3." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1314-1324. 2019.
> Ma, Ningning, Xiangyu Zhang, Hai-Tao Zheng, and Jian Sun. "Shufflenet v2: Practical guidelines for efficient cnn architecture design." In Proceedings of the European conference on computer vision (ECCV), pp. 116-131. 2018.
> Yu, Changqian, Bin Xiao, Changxin Gao, Lu Yuan, Lei Zhang, Nong Sang, and Jingdong Wang. "Lite-hrnet: A lightweight high-resolution network." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10440-10450. 2021.
> Han, Kai, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, and Chang Xu. "Ghostnet: More features from cheap operations." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1580-1589. 2020.

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|MobileSeg|mobilenetv2|1024x512|80000|%|%|%|[model]() \| [log]() \| [vdl]()|
|MobileSeg|mobilenetv3|1024x512|80000|%|%|%|[model]() \| [log]() \| [vdl]()|
|MobileSeg|shufflenetv2|1024x512|80000|%|%|%|[model]() \| [log]() \| [vdl]()|
|MobileSeg|litehrnet18|1024x512|80000|%|%|%|[model]() \| [log]() \| [vdl]()|
|MobileSeg|ghostnet|1024x512|80000|%|%|%|[model]() \| [log]() \| [vdl]()|

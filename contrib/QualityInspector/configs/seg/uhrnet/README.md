# U-HRNet: Delving into Improving Semantic Representation of High Resolution Network for Dense Prediction

## Reference
> Jian Wang, Xiang Long, Guowei Chen, Zewu Wu, Zeyu Chen, Errui Ding et al. "U-HRNet: Delving into Improving Semantic Representation of High Resolution Network for Dense Prediction" arXiv preprint arXiv:2210.07140 (2022).

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|FCN|UHRNet_W18_small|1024x512|80000|77.66%|78.26%|78.47%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fcn_uhrnetw18_small_cityscapes_1024x512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fcn_uhrnetw18_small_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=9cb0e961bc1f89d3484190f9d4de550b)|
|FCN|UHRNet_W18_small|1024x512|120000|78.39%|79.09%|79.03%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fcn_uhrnetw18_small_cityscapes_1024x512_120k_bs3/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fcn_uhrnetw18_small_cityscapes_1024x512_120k_bs3/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=6f6c41e46cf8b26d3a941bf7e09698f8)|
|FCN|UHRNet_W48|1024x512|80000|81.28%|81.76%|81.48%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fcn_uhrnetw48_cityscapes_1024x512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fcn_uhrnetw48_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=1c2fbc3a5558d530c2a1fc8c2cd34da5)|
|FCN|UHRNet_W48|1024x512|120000|81.91%|82.39%|82.28%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fcn_uhrnetw48_cityscapes_1024x512_120k_bs3/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fcn_uhrnetw48_cityscapes_1024x512_120k_bs3/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=a94e548519f9487c435530532f7a027c)|

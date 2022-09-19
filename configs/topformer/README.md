# TopFormer: Token Pyramid Transformer for Mobile Semantic Segmentation

## Reference

> Zhang, Wenqiang, Zilong Huang, Guozhong Luo, Tao Chen, Xinggang Wang, Wenyu Liu, Gang Yu,and Chunhua Shen. "TopFormer: Token Pyramid Transformer for Mobile Semantic Segmentation." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12083-12093. 2022.


## Performance

### ADE20k

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|---|---|---|---|---|---|---|---|
|TopFormer-Base |topformer|512x512|160000| 38.28% | 38.59% | - |[model](https://paddleseg.bj.bcebos.com/dygraph/ade20k/topformer_base_ade20k_512x512_160k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/ade20k/topformer_base_ade20k_512x512_160k/log_train.txt) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=1530ee894b33a363677472fdcae5d13a) |
|TopFormer-Small|topformer|512x512|160000| 35.60% | 35.83% | - |[model](https://paddleseg.bj.bcebos.com/dygraph/ade20k/topformer_small_ade20k_512x512_160k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/ade20k/topformer_small_ade20k_512x512_160k/log_train.txt) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=c6070db4366510a20d47fb4645797a27) |
|TopFormer-Tiny |topformer|512x512|160000| 32.49% | 32.75% | - |[model](https://paddleseg.bj.bcebos.com/dygraph/ade20k/topformer_tiny_ade20k_512x512_160k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/ade20k/topformer_tiny_ade20k_512x512_160k/log_train.txt) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=462723f835db022d3eba1b4db87350e3) |


Note that, the input resulution of TopFormer should be a multiple of 32.

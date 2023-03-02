# K-Net: Towards Unified Image Segmentation


## Reference

> TWenwei Zhang, Jiangmiao Pang, Kai Chen, Chen Change Loy. "K-Net: Towards Unified Image Segmentation." In: Proc. Advances in Neural Information Processing Systems (NeurIPS), 2021.

## Performance

### ADE20K

| Method          | Backbone   |    Resolution | Training Iters | mIoU   | Links                                                                                                           |
|-----------------|------------|-----|---------|--------|-----------------------------------------------------------------------------------------------------------------|
| K-Net + UPerNet | ResNet50   | 512x512    | 80000   | 44.03% | [model](https://paddleseg.bj.bcebos.com/dygraph/ade20k/knet_s3_upernet_resnet50_ade20k_512x512_80k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/ade20k/knet_s3_upernet_resnet50_ade20k_512x512_80k/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/index?id=b5277ff16b7f96867b748ebe48237690)|
| K-Net + FCN     | ResNet50   | 512x512    | 80000   | 43.31% | [model](https://paddleseg.bj.bcebos.com/dygraph/ade20k/knet_s3_fcn_resnet50_ade20k_512x512_80k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/ade20k/knet_s3_fcn_resnet50_ade20k_512x512_80k/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/index?id=8a3b9360be5dd959f1b7135ba9ef2110)|

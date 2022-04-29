# FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation

## Reference
> Wu, Huikai, Junge Zhang, Kaiqi Huang, Kongming Liang, and Yizhou Yu. "Fastfcn: Rethinking dilated convolution in the backbone for semantic segmentation." arXiv preprint arXiv:1903.11816 (2019).

## Performance

### ADE20K

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|FastFCN|ResNet50_vd|480x480|120000|43.76%|44.11%|44.48%|[model](https://bj.bcebos.com/paddleseg/dygraph/ade20k/fastfcn_resnet50_os8_ade20k_480x480_120k/model.pdparams) \|[log](https://bj.bcebos.com/paddleseg/dygraph/ade20k/fastfcn_resnet50_os8_ade20k_480x480_120k/train.log)\|[vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=e159d5be3860b8d08762c0416ab54acc)|

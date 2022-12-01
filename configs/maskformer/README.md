# Per-Pixel Classification is Not All You Need for Semantic Segmentation

## Reference

> Cheng, Bowen, Alex Schwing, and Alexander Kirillov. "Per-pixel classification is not all you need for semantic segmentation." Advances in Neural Information Processing Systems 34 (2021): 17864-17875.

## Performance

### ADE20k

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Maskformer|SwinTransformer|512x512|160000|-|-|-|[model](https://bj.bcebos.com/paddleseg/dygraph/ade20k/maskformer_ade20k_swin_tiny/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/ade20k/maskformer_ade20k_swin_tiny/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=894b6404bb93abf5d755b2a22aba0ade)|

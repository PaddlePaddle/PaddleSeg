# Rethinking Vision Transformers for MobileNet Size and Speed

## Reference

> Yanyu Li, Ju Hu, Yang Wen, Georgios Evangelidis and Kamyar Salahi. "Rethinking Vision Transformers for MobileNet Size and Speed." arXiv:2212.08059.

## Performance

### ADE20k

|       Model       | size |  Head  | Resolution | Training Iters | mIoU (slice) |              Links              |
| :---------------: | :--: | :----: | :--------: | :------------: | :----------: | :-----------------------------: |
| EfficientFormerV2 |  s2  | FPNNet |  512x512   |     40000      |    42.16     | [model](https://bj.bcebos.com/paddleseg/dygraph/ade20k/efficientformerv2_s2_ade20k_512x512_40k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/ade20k/efficientformerv2_s2_ade20k_512x512_40k/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=7397511b1f7607510aaafc218e4ec90d) |

# HRFormer: High-Resolution Transformer for Dense Prediction

## Reference

> Yuhui Yuan, Rao Fu, Lang Huang, Weihong Lin, Chao Zhang, Xilin Chen, and Jingdong Wang. "HRFormer: High-Resolution Transformer for Dense Prediction." arXiv preprint arXiv:2110.09408v2 (2021).

## Performance

### CityScapes

| Model  |    Backbone    | Resolution | Training Iters |  mIoU  | mIoU (flip) | mIoU (ms+flip) |                            Links                             |
| :----: | :------------: | :--------: | :------------: | :----: | :---------: | :------------: | :----------------------------------------------------------: |
| OCRNet | HRformer_small |  1024x512  |     80000      | 80.62% |   80.82%    |     80.98%     | [model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/ocrnet_hrformer_small_cityscapes_1024x512_80k_ce_ohem/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/ocrnet_hrformer_small_cityscapes_1024x512_80k_ce_ohem/train.log) \| [vdl]() |
| OCRNet | HRFormer_base  |  1024x512  |     80000      | 80.35% |   80.63%    |     80.87%     | [model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/ocrnet_hrformer_base_cityscapes_1024x512_80k_ce_ohem/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/ocrnet_hrformer_base_cityscapes_1024x512_80k_ce_ohem/train.log) \| [vdl]() |

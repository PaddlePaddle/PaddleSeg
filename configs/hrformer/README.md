# HRFormer: High-Resolution Transformer for Dense Prediction

## Reference

> Yuhui Yuan, Rao Fu, Lang Huang, Weihong Lin, Chao Zhang, Xilin Chen, and Jingdong Wang. "HRFormer: High-Resolution Transformer for Dense Prediction." arXiv preprint arXiv:2110.09408v2 (2021).

## Performance

### CityScapes

| Model  |    Backbone    | Resolution | Training Iters |  mIoU  | mIoU (flip) | mIoU (ms+flip) |                            Links                             |
| :----: | :------------: | :--------: | :------------: | :----: | :---------: | :------------: | :----------------------------------------------------------: |
| OCRNet | HRformer_small |  1024x512  |     80000      | 80.62% |   80.82%    |     80.98%     | [model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/ocrnet_hrformer_small_cityscapes_1024x512_80k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/ocrnet_hrformer_small_cityscapes_1024x512_80k/train.log) |
| OCRNet | HRFormer_base  |  1024x512  |     80000      | 80.35% |   80.63%    |     80.87%     | [model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/ocrnet_hrformer_base_cityscapes_1024x512_80k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/ocrnet_hrformer_base_cityscapes_1024x512_80k/train.log) |

&emsp;   The accuracy obtained by the model using `HRFormer_base` as backbone is lower than that in the original paper. We attribute this performance gap to the difference in `OCRNet` specification. In the original implementation, the authors fixed the number of `hidden channels` of `aux_head` to 512. Yet, in our `OCRNet` implementation,  the number of `hidden channels` of `aux_head` is equal to `input_channel`.

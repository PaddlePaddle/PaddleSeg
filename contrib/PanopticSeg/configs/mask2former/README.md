# Masked-Attention Mask Transformer for Universal Image Segmentation

## Reference

> Cheng, Bowen, et al. "Masked-attention mask transformer for universal image segmentation." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2022.

## Prerequesites

+ Compile and install external operators by executing the following instructions:

```shell
cd paddlepanseg/models/ops
cd ms_deform_attn
python setup.py install
```

## Performance

### COCO minival

| Model | Backbone | Resolution | Training Iters | PQ | mIoU | mAP50 | Links |
|-|-|-|-|-|-|-|-|
|Mask2Former|ResNet50-vd|1024x1024|370k|53.69%|63.22%|49.16%|[config](mask2former_resnet50_os16_coco_1024x1024_bs4_370k.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/panoptic_segmentation/coco/mask2former_resnet50_os16_coco_1024x1024_bs4_370k/model.pdparams)|

+ *The models were trained using 4 GPUs.*

## Export

Currently, this model can **NOT** be exported.

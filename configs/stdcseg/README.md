# Rethinking BiSeNet For Real-time Semantic Segmentation

## Reference

> Fan, Mingyuan, Shenqi Lai, Junshi Huang, Xiaoming Wei, Zhenhua Chai, Junfeng Luo, and Xiaolin Wei. "Rethinking BiSeNet For Real-time Semantic Segmentation." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9716-9725. 2021.


## Performance

### CityScapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|---|---|---|---|---|---|---|---|
|STDC1-Seg50|STDC1|1024x512|80000|74.74%|75.71%|76.64%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/stdc1_seg_cityscapes_1024x512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/stdc1_seg_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=f450fdef2b3b02574e3eb293242d1fbd) |
|STDC2-Seg50|STDC2|1024x512|80000|77.60%|78.32%|78.83%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/stdc2_seg_cityscapes_1024x512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/stdc2_seg_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=1f9c39f10e94327803faae96a516a7a6) |

# Object-Contextual Representations for Semantic Segmentation

## 模型介绍

    @article{YuanCW20,
        title={Object-Contextual Representations for Semantic Segmentation},
        author={Yuhui Yuan and Xilin Chen and Jingdong Wang},
        booktitle={ECCV},
        year={2020}
    }

## 模型效果

### CityScapes

|Model|Backbone|Training Iters|mIoU|mIoU(ms+flip)|Link|
|-|-|-|-|-|-|
|OCRNet|HRNet_w18|160000|82.15%||[training log](https://paddleseg.bj.bcebos.com/dygraph/ocrnet/hrnetw18/train.log) \| [vdl log](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=176bf6ca4d89957ffe62ac7c30fcd039) \| [model](https://paddleseg.bj.bcebos.com/dygraph/ocrnet/hrnetw18/model.pdparams) \| [training command](https://paddleseg.bj.bcebos.com/dygraph/ocrnet/hrnetw18/train.sh)|
|OCRNet|HRNet_w48|160000|80.67%||[training log]() \| [vdl log](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=901a5d0a78b71ca56f06002f05547837) \| [model]() \| [training command]()|

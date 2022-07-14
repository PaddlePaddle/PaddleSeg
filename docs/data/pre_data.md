[简体中文](./pre_data_cn.md) | English

# Prepare Public Dataset

For public dataset, you only need to download and save it in specific directory, and then use PaddleSeg to train model.

## Save Directory

PaddleSeg defines the dataset path in config files according the following directory.

It is recommended to download and save public dataset in `PaddleSeg/data`.

```
PaddleSeg
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── ADEChallengeData2016
│   │   │── annotations
│   │   │   ├── training
│   │   │   ├── validation
│   │   │── images
│   │   │   ├── training
│   │   │   ├── validation
│   ├── VOCdevkit
│   │   ├── VOC2012
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClass
│   │   │   ├── SegmentationClassAug
│   │   │   ├── ImageSets
│   │   │   │   ├── Segmentation
```

## Download Public Dataset

Download Cityscapes:
* [Cityscapes Official Website](https://www.cityscapes-dataset.com/login/)
* [BaiDuYun](https://paddleseg.bj.bcebos.com/dataset/cityscapes.tar)
* `wget https://paddleseg.bj.bcebos.com/dataset/cityscapes.tar`

Download ADE20K：
* [ADE20K Official Website](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
* [BaiDuYun](https://paddleseg.bj.bcebos.com/dataset/ADEChallengeData2016.zip)
* `wget https://paddleseg.bj.bcebos.com/dataset/ADEChallengeData2016.zip`

简体中文 | [English](./pre_data.md)

# 准备公开数据集

对于公开数据集，大家只需要下载并存放到特定目录，就可以使用PaddleSeg进行模型训练评估。

## 公开数据集存放目录

PaddleSeg提供的配置文件，是按照如下存放目录，来定义公开数据集的路径。

所以，建议大家下载公开数据集，然后存放到`PaddleSeg/data`目录下。

如果公开数据集不是按照如下目录进行存放，大家需要根据实际情况，手动修改配置文件中的数据集目录。

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

## 公开数据集下载链接

Cityscapes下载方式:
* [Cityscapes官网](https://www.cityscapes-dataset.com/login/)
* [百度云](https://paddleseg.bj.bcebos.com/dataset/cityscapes.tar)
* `wget https://paddleseg.bj.bcebos.com/dataset/cityscapes.tar`


ADE20K下载：
* [ADE20K官网](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
* [百度云](https://paddleseg.bj.bcebos.com/dataset/ADEChallengeData2016.zip)
* `wget https://paddleseg.bj.bcebos.com/dataset/ADEChallengeData2016.zip`

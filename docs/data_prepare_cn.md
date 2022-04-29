简体中文 | [English](data_prepare.md)

# 数据集准备

PaddleSeg目前支持CityScapes、ADE20K、Pascal VOC等数据集的加载，在加载数据集时，如若本地不存在对应数据，则会自动触发下载(除Cityscapes数据集).

## CityScapes数据集
Cityscapes是关于城市街道场景的语义理解图片数据集。它主要包含来自50个不同城市的街道场景，
拥有5000张（2048 x 1024）城市驾驶场景的高质量像素级注释图像，包含19个类别。其中训练集2975张， 验证集500张和测试集1525张。

由于协议限制，请自行前往[CityScapes官网](https://www.cityscapes-dataset.com/)下载数据集，
我们建议您将数据集存放于`PaddleSeg/data`中，以便与我们配置文件完全兼容。数据集下载后请组织成如下结构：

```
    cityscapes
    |
    |--leftImg8bit
    |  |--train
    |  |--val
    |  |--test
    |
    |--gtFine
    |  |--train
    |  |--val
    |  |--test
```

运行下列命令进行标签转换：
```shell
pip install cityscapesscripts
python tools/convert_cityscapes.py --cityscapes_path data/cityscapes --num_workers 8
```
其中`cityscapes_path`应根据实际数据集路径进行调整。 `num_workers`决定启动的进程数，可根据实际情况进行调整大小。

## Pascal VOC 2012数据集
[Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/)数据集以对象分割为主，包含20个类别和背景类，其中训练集1464张，验证集1449张。
通常情况下会利用[SBD(Semantic Boundaries Dataset)](http://home.bharathh.info/pubs/codes/SBD/download.html)进行扩充，扩充后训练集10582张。
运行下列命令进行SBD数据集下载并进行扩充：
```shell
python tools/voc_augment.py --voc_path data/VOCdevkit --num_workers 8
```
其中`voc_path`应根据实际数据集路径进行调整。

**注意** 运行前请确保在PaddleSeg目录下执行过下列命令：
```shell
export PYTHONPATH=`pwd`
# windows下请执行下面的命令
# set PYTHONPATH=%cd%
```

## ADE20K数据集
[ADE20K](http://sceneparsing.csail.mit.edu/)由MIT发布的可用于场景感知、分割和多物体识别等多种任务的数据集。
其涵盖了150个语义类别，包括训练集20210张，验证集2000张。

## Coco Stuff数据集
Coco Stuff是基于Coco数据集的像素级别语义分割数据集。它主要覆盖172个类别，包含80个'thing'，91个'stuff'和1个'unlabeled',我们忽略'unlabeled'类别，并将其index设为255，不记录损失。因此提供的训练版本为171个类别。其中，训练集118k, 验证集5k.

在使用Coco Stuff数据集前， 请自行前往[COCO-Stuff主页](https://github.com/nightrome/cocostuff)下载数据集，或者下载[coco2017训练集原图](http://images.cocodataset.org/zips/train2017.zip), [coco2017验证集原图](http://images.cocodataset.org/zips/val2017.zip)及[标注图](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip)
我们建议您将数据集存放于`PaddleSeg/data`中，以便与我们配置文件完全兼容。数据集下载后请组织成如下结构：

```
    cocostuff
    |
    |--images
    |  |--train2017
    |  |--val2017
    |
    |--annotations
    |  |--train2017
    |  |--val2017
```  


运行下列命令进行标签转换：

```shell
python tools/convert_cocostuff.py --annotation_path /PATH/TO/ANNOTATIONS --save_path /PATH/TO/CONVERT_ANNOTATIONS
```
其中`annotation_path`应根据下载cocostuff/annotations文件夹的实际路径填写。 `save_path`决定转换后标签的存放位置。


其中，标注图像的标签从0,1依次取值，不可间隔。若有需要忽略的像素，则按255进行标注。

## Pascal Context数据集
Pascal Context是基于PASCAL VOC 2010数据集额外标注的像素级别的语义分割数据集。我们提供的转换脚本支持60个类别，index为0是背景类别。该数据集中训练集4996, 验证集5104张.


在使用Pascal Context数据集前， 请先下载[VOC2010](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar)，随后自行前往[Pascal-Context主页](https://www.cs.stanford.edu/~roozbeh/pascal-context/)下载数据集及[标注](https://codalabuser.blob.core.windows.net/public/trainval_merged.json)
我们建议您将数据集存放于`PaddleSeg/data`中，以便与我们配置文件完全兼容。数据集下载后请组织成如下结构：

```
    VOC2010
    |
    |--Annotations
    |
    |--ImageSets
    |
    |--SegmentationClass
    |  
    |--JPEGImages
    |
    |--SegmentationObject
    |
    |--trainval_merged.json
```

运行下列命令进行标签转换：

```shell
python tools/convert_voc2010.py --voc_path /PATH/TO/VOC ----annotation_path /PATH/TO/JSON
```
其中`voc_path`应根据下载VOC2010文件夹的实际路径填写。 `annotation_path`决定下载trainval_merged.json的存放位置。



其中，标注图像的标签从0，1，2依次取值，不可间隔。若有需要忽略的像素，则按255(默认的忽略值）进行标注。在使用Pascal Context数据集时，需要安装[Detail](https://github.com/zhanghang1989/detail-api).

## 自定义数据集

如果您需要使用自定义数据集进行训练，请按照以下步骤准备数据.

1.推荐整理成如下结构

```
    custom_dataset
        |
        |--images
        |  |--image1.jpg
        |  |--image2.jpg
        |  |--...
        |
        |--labels
        |  |--label1.jpg
        |  |--label2.png
        |  |--...
        |
        |--train.txt
        |
        |--val.txt
        |
        |--test.txt
```

其中train.txt和val.txt的内容如下所示：

```
    images/image1.jpg labels/label1.png
    images/image2.jpg labels/label2.png
    ...
```

2.标注图像的标签从0,1依次取值，不可间隔。若有需要忽略的像素，则按255进行标注。

可按如下方式对自定义数据集进行配置：
```yaml
train_dataset:
  type: Dataset
  dataset_root: custom_dataset
  train_path: custom_dataset/train.txt
  num_classes: 2
  transforms:
    - type: ResizeStepScaling
      min_scale_factor: 0.5
      max_scale_factor: 2.0
      scale_step_size: 0.25
    - type: RandomPaddingCrop
      crop_size: [512, 512]
    - type: RandomHorizontalFlip
    - type: Normalize
  mode: train
```

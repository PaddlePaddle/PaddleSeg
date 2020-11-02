# 数据集准备

PaddleSeg目前支持CityScapes、ADE20K、Pascal VOC等数据集的加载，在加载数据集时，如若本地不存在对应数据，则会自动触发下载(除Cityscapes数据集).

## 关于CityScapes数据集
Cityscapes是关于城市街道场景的语义理解图片数据集。它主要包含来自50个不同城市的街道场景，
拥有5000张（2048 x 1024）城市驾驶场景的高质量像素级注释图像，包含19个类别。其中训练集2975张， 验证集500张和测试集1525张。

由于协议限制，请自行前往[CityScapes官网](https://www.cityscapes-dataset.com/)下载数据集，
我们建议您将数据集存放于`PaddleSeg/dygraph/data`中，以便与我们配置文件完全兼容。数据集下载后请组织成如下结构：

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

运行下列命令进行标签转换：
```shell
pip install cityscapesscripts
python tools/convert_cityscapes.py --cityscapes_path data/cityscapes --num_workers 8
```
其中`cityscapes_path`应根据实际数据集路径进行调整。 `num_workers`决定启动的进程数，可根据实际情况进行调整大小。

## 关于Pascal VOC 2012数据集
[Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/)数据集以对象分割为主，包含20个类别和背景类，其中训练集1464张，验证集1449张。
通常情况下会利用[SBD(Semantic Boundaries Dataset)](http://home.bharathh.info/pubs/codes/SBD/download.html)进行扩充，扩充后训练集10582张。
运行下列命令进行SBD数据集下载并进行扩充：
```shell
python tools/voc_augment.py --voc_path data/VOCdevkit --num_workers 8
```
其中`voc_path`应根据实际数据集路径进行调整。

**注意** 运行前请确保在dygraph目录下执行过下列命令：
```shell
export PYTHONPATH=`pwd`
# windows下请执行相面的命令
# set PYTHONPATH=%cd%
```

## 关于ADE20K数据集
[ADE20K](http://sceneparsing.csail.mit.edu/)由MIT发布的可用于场景感知、分割和多物体识别等多种任务的数据集。
其涵盖了150个语义类别，包括训练集20210张，验证集2000张。

## 自定义数据集

如果您需要使用自定义数据集进行训练，请按照以下步骤准备数据.

1.推荐整理成如下结构

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

其中train.txt和val.txt的内容如下所示：

    images/image1.jpg labels/label1.png
    images/image2.jpg labels/label2.png
    ...

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

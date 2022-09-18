简体中文 | [English](./pre_data.md)

# 准备公开数据集

对于公开数据集，大家只需要下载并存放到特定目录，就可以使用PaddleSeg进行模型训练评估。

## 公开数据集存放目录

PaddleSeg是按照如下数据集存放目录，来定义配置文件中默认的公开数据集路径。
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

## 公开数据集下载

### CityScapes数据集

Cityscapes是关于城市街道场景的语义理解图片数据集。它主要包含来自50个不同城市的街道场景，拥有5000张（2048 x 1024）高质量像素级注释图像，包含19个类别。Cityscapes数据集的训练集2975张，验证集500张，测试集1525张。

请前往[CityScapes官网](https://www.cityscapes-dataset.com/)下载数据集。
数据集结构如下：

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

下载原始数据集后，运行下面命令进行转换，其中`cityscapes_path`是数据集保存的根目录，`num_workers`是进程数。执行完成后，转换后的数据集依旧保存在原先数据集目录下。

```shell
pip install cityscapesscripts
python tools/convert_cityscapes.py --cityscapes_path data/cityscapes --num_workers 8
```

### ADE20K数据集

ADE20K数据集由MIT发布的可用于场景感知、分割和多物体识别等多种任务的数据集，其涵盖了150个语义类别，包括训练集20210张，验证集2000张。

大家可以到[官方网站](https://groups.csail.mit.edu/vision/datasets/ADE20K/)下载该数据集。

### Pascal VOC 2012数据集

Pascal VOC 2012数据集以对象分割为主，包含20个类别和背景类，其中训练集1464张，验证集1449张。

大家可以到[官方网站](http://host.robots.ox.ac.uk/pascal/VOC/)下载该数据集。

通常情况下，大家会利用[SBD(Semantic Boundaries Dataset)](http://home.bharathh.info/pubs/codes/SBD/download.html)对VOC 2012数据集进行扩充，得到的训练集是10582张。

运行下列命令进行SBD数据集进行扩充，其中`voc_path`应根据实际数据集路径进行设置。

```shell
cd PaddleSeg
python tools/voc_augment.py --voc_path data/VOCdevkit --num_workers 8
```

### Coco Stuff数据集

Coco Stuff是基于Coco数据集的像素级别语义分割数据集。它主要覆盖172个类别，包含80个'thing'，91个'stuff'和1个'unlabeled'，我们忽略'unlabeled'类别，并将其index设为255，不记录损失。因此提供的训练版本为171个类别。其中，训练集118k, 验证集5k。

在使用Coco Stuff数据集前， 请自行前往[COCO-Stuff主页](https://github.com/nightrome/cocostuff)下载数据集，或者下载[coco2017训练集原图](http://images.cocodataset.org/zips/train2017.zip), [coco2017验证集原图](http://images.cocodataset.org/zips/val2017.zip)及[标注图](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip)。

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


运行下列命令进行标签转换，其中`annotation_path`应根据下载cocostuff/annotations文件夹的实际路径填写，`save_path`决定转换后标签的存放位置。

```shell
python tools/convert_cocostuff.py --annotation_path /PATH/TO/ANNOTATIONS --save_path /PATH/TO/CONVERT_ANNOTATIONS
```


## Pascal Context数据集

Pascal Context是基于PASCAL VOC 2010数据集额外标注的像素级别的语义分割数据集。我们提供的转换脚本支持60个类别，index为0是背景类别。该数据集中训练集4996, 验证集5104张.

在使用Pascal Context数据集前， 请先下载[VOC2010](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar)，随后自行前往[Pascal-Context主页](https://www.cs.stanford.edu/~roozbeh/pascal-context/)下载数据集及[标注](https://codalabuser.blob.core.windows.net/public/trainval_merged.json)。

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


其中，标注图像的标签从0，1，2依次取值，不可间隔。若有需要忽略的像素，则按255(默认的忽略值）进行标注。在使用Pascal Context数据集时，需要安装[Detail](https://github.com/zhanghang1989/detail-api)。

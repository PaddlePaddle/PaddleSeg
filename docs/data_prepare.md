English | [简体中文](data_prepare_cn.md)

# Dataset Preparation

PaddleSeg currently supports the dataset of CityScapes, ADE20K, Pascal VOC and so on.
When loading a dataset, the system automatically triggers the download (except for Cityscapes dataset) if the data does not exist locally.

## CityScapes Dataset
Cityscapes is a dataset of semantically understood images of urban street scenes. It mainly contains street scenes from 50 different cities, with 5000 (2048 x 1024) high quality pixel-level annotated images of urban driving scenes. It contains 19 categories. There are 2975 training sets, 500 validation sets and 1525 test sets.

Due to restrictions, please visit [CityScapes website](https://www.cityscapes-dataset.com/)to download dataset.
We recommend that you store dataset in `PaddleSeg/data` for full compatibility with our config files. Please organize the dataset into the following structure after downloading:

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

Run the following command to convert labels:
```shell
pip install cityscapesscripts
python tools/convert_cityscapes.py --cityscapes_path data/cityscapes --num_workers 8
```
where `cityscapes_path` should be adjusted according to the actual dataset path. `num_workers` determines the number of processes to be started. The value can be adjusted as required.

## Pascal VOC 2012 dataset
[Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/) is mainly object segmentation, including 20 categories and background classes, including 1464 training sets and 1449 validation sets.
Generally, we will use [SBD(Semantic Boundaries Dataset)](http://home.bharathh.info/pubs/codes/SBD/download.html) to expand the dataset. Theer are 10582 training sets after expanding.
Run the following commands to download the SBD dataset and use it to expand:
```shell
python tools/voc_augment.py --voc_path data/VOCdevkit --num_workers 8
```
where `voc_path`should be adjusted according to the actual dataset path.

**Note** Before running, make sure you have executed the following commands in the PaddleSeg directory:
```shell
export PYTHONPATH=`pwd`
# In Windows, run the following command
# set PYTHONPATH=%cd%
```

## ADE20K Dataset
[ADE20K](http://sceneparsing.csail.mit.edu/) published by MIT that can be used for a variety of tasks such as scene perception, segmentation, and multi-object recognition.
It covers 150 semantic categories, including 20210 training sets and 2000 validation sets.

## Coco Stuff Dataset
Coco Stuff is a pixel-level semantically segmented dataset based on Coco datasets. It covers 172 catefories, including 80 'thing' classes, 91 'stuff' classes amd one 'unlabeled' classes. 'unlabeled' is ignored and the index is set to 255 which has not contribution to loss. The training version is therefore provided in 171 categories. There are 118k training sets, 5k validation sets.

Before using Coco Stuff dataset， please go to [COCO-Stuff website](https://github.com/nightrome/cocostuff) to download dataset or download [coco2017 training sets with origin images](http://images.cocodataset.org/zips/train2017.zip), [coco2017 validation sets with origin images](http://images.cocodataset.org/zips/val2017.zip) and [annotations images](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip)
We recommend that you store dataset in `PaddleSeg/data` for full compatibility with our config files. Please organize the dataset into the following structure after downloading:

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

Run the following command to convert labels:

```shell
python tools/convert_cocostuff.py --annotation_path /PATH/TO/ANNOTATIONS --save_path /PATH/TO/CONVERT_ANNOTATIONS
```
where `annotation_path` should be filled according to the `cocostuff/annotations` actual path. `save_path` determines the location of the converted label.

Where, the labels of the labeled images are taken in sequence from 0, 1, ... and cannot be separated. If there are pixels that need to be ignored, they should be labeled to 255.

## Pascal Context Dataset
Pascal Context is a pixel-level semantically segmented dataset based on the Pascal VOC 2010 dataset with additional annotations. The conversion script we provide supports 60 categories, with index 0 being the background category. There are 4996 training sets and 5104 verification sets in this dataset.


Before using Pascal Context dataset， Please download [VOC2010](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar) firstly，then go to [Pascal-Context home page](https://www.cs.stanford.edu/~roozbeh/pascal-context/)to download dataset and [annotations](https://codalabuser.blob.core.windows.net/public/trainval_merged.json)
We recommend that you store dataset in `PaddleSeg/data` for full compatibility with our config files. Please organize the dataset into the following structure after downloading:

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

Run the following command to convert labels:

```shell
python tools/convert_voc2010.py --voc_path /PATH/TO/VOC ----annotation_path /PATH/TO/JSON
```
where `voc_path` should be filled according to the voc2010 actual path. `annotation_path` is the trainval_merged.json saved path.


Where, the labels of the labeled images are taken in sequence from 0, 1, 2, ... and cannot be separated. If there are pixels that need to be ignored, they should be labeled to 255 (default ignored value). When using Pascal Context dataset, [Detail](https://github.com/zhanghang1989/detail-api) need to be installed.


## Custom Dataset

If you need to use a custom dataset for training, prepare the data as following steps.

1.The following structure is recommended

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

The content of train.txt and val.txt is as following：

```
    images/image1.jpg labels/label1.png
    images/image2.jpg labels/label2.png
    ...
```

2.The labels of the labeled images are taken in sequence from 0, 1, ... and cannot be separated. If there are pixels that need to be ignored, they should be labeled to 255.

You can configure a custom dataset as following：
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

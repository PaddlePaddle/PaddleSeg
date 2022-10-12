English | [简体中文](datasets_cn.md)
# paddleseg.datasets
- [Custom Dataset](#custom-dataset)
- [Cityscapes](#Cityscapes)
- [PascalVOC](#PascalVOC)
- [ADE20K](#ADE20K)
- [OpticDiscSeg](#OpticDiscSeg)

## [Custom Dataset](../../../paddleseg/datasets/dataset.py)
```python
class paddleseg.datasets.Dataset(transforms, dataset_root, num_classes, mode='train', train_path=None, val_path=None, test_path=None, separator=' ', ignore_index=255, edge=False)
```
    Pass in a custom dataset that conforms to the format.

### Args
* **transforms** (list): Transforms for image.
* **dataset_root** (str): The dataset directory.
* **num_classes** (int): Number of classes.
* **mode** (str, optional): which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
* **train_path** (str, optional): The train dataset file. When mode is 'train', train_path is necessary.
        The contents of train_path file are as follow:
        image1.jpg ground_truth1.png
        image2.jpg ground_truth2.png
* **val_path** (str, optional): The evaluation dataset file. When mode is 'val', val_path is necessary.
        The contents is the same as train_path
* **test_path** (str, optional): The test dataset file. When mode is 'test', test_path is necessary.
        The annotation file is not necessary in test_path file.
* **separator** (str, optional): The separator of dataset list. Default: ' '.
* **edge** (bool, optional): Whether to compute edge while training. Default: False

### Examples

```python
import paddleseg.transforms as T
from paddleseg.datasets import Dataset

transforms = [T.RandomPaddingCrop(crop_size=(512,512)), T.Normalize()]
dataset_root = 'dataset_root_path'
train_path = 'train_path'
num_classes = 2
dataset = Dataset(transforms = transforms,
                  dataset_root = dataset_root,
                  num_classes = 2,
                  train_path = train_path,
                  mode = 'train')
```

## [Cityscapes](../../../paddleseg/datasets/cityscapes.py)
```python
class paddleseg.datasets.Cityscapes(transforms, dataset_root, mode='train', edge=False)
```
    Cityscapes dataset `https://www.cityscapes-dataset.com/`.
    The folder structure is as follow:

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

    Make sure there are **labelTrainIds.png in gtFine directory. If not, please run the conver_cityscapes.py in tools.

### Args
* **transforms** (list): Transforms for image.
* **dataset_root** (str): Cityscapes dataset directory.
* **mode** (str, optional): Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
* **edge** (bool, optional): Whether to compute edge while training. Default: False


## [PascalVOC](../../../paddleseg/datasets/voc.py)
```python
class paddleseg.datasets.PascalVOC(transforms, dataset_root=None, mode='train', edge=False)
```
    PascalVOC2012 dataset `http://host.robots.ox.ac.uk/pascal/VOC/`.
    If you want to augment the dataset, please run the voc_augment.py in tools/data.

### Args
* **transforms** (list): Transforms for image.
* **dataset_root** (str): The dataset directory. Default: None
* **mode** (str, optional): Which part of dataset to use. it is one of ('train', 'trainval', 'trainaug', 'val').
        If you want to set mode to 'trainaug', please make sure the dataset have been augmented. Default: 'train'.
* **edge** (bool, optional): Whether to compute edge while training. Default: False

## [ADE20K](../../../paddleseg/datasets/ade.py)
```python
class paddleseg.datasets.ADE20K(transforms, dataset_root=None, mode='train', edge=False)
```
    ADE20K dataset `http://sceneparsing.csail.mit.edu/`.

### Args
* **transforms** (list): A list of image transformations.
* **dataset_root** (str, optional): The ADK20K dataset directory. Default: None.
* **mode** (str, optional): A subset of the entire dataset. It should be one of ('train', 'val'). Default: 'train'.
* **edge** (bool, optional): Whether to compute edge while training. Default: False

## [OpticDiscSeg](../../../paddleseg/datasets/optic_disc_seg.py)
```python
class paddleseg.datasets.OpticDiscSeg(dataset_root=None, transforms=None, mode='train', edge=False)
```
    OpticDiscSeg dataset is extraced from iChallenge-AMD `https://ai.baidu.com/broad/subordinate?dataset=amd`.

### Args
* **transforms** (list): Transforms for image.
* **dataset_root** (str): The dataset directory. Default: None
* **mode** (str, optional): Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
* **edge** (bool, optional): Whether to compute edge while training. Default: False

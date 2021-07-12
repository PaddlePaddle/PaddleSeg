简体中文 | [English](datasets.md)
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
    传入符合格式的自定义数据集。


### 参数
* **transforms** (list): 对图像的变换方法。
* **dataset_root** (str): 数据集存放目录。
* **num_classes** (int): 像素类别数目。
* **mode** (str, optional): 使用何种数据集。应为('train', 'val', 'test')中的一种。默认: 'train'。
* **train_path** (str, optional): 训练数据集的路径。当模式指定为'train'时, 必须指定 train_path 参数。train_path 文件的内容格式如下：
    >image1.jpg ground_truth1.png
    >image2.jpg ground_truth2.png
* **val_path** (str, optional): 验证数据集的路径。当模式指定为'val'时, 必须指定 val_path 参数。
        val_path 文件的内容格式请参照 train_path 文件。
* **test_path** (str, optional): 测试数据集的路径。当模式指定为'test'时,必须指定 test_path 参数。
        在test_path 文件中，annotation file （标注文件）不是必须的。
* **separator** (str, optional): 数据集列表的分割符。默认: ' '。
* **edge** (bool, optional): 训练时是否指定求取边缘。默认: False。

### 举例

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
    Cityscapes 数据集 `https://www.cityscapes-dataset.com/`。
    其文件结构如下所示：

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
    请确保目录 gtFine 中存在 **labelTrainIds.png。如果不存在, 请运行 tools 中的conver_cityscapes.py。

### 参数
* **transforms** (list): 对图像的变换方法。
* **dataset_root** (str): Cityscapes 数据集存放目录。
* **mode** (str, optional): 使用何种数据集。应为('train', 'val', 'test')中的一种。默认: 'train'。
* **edge** (bool, optional): 训练时是否指定求取边缘。默认: False。


## [PascalVOC](../../../paddleseg/datasets/voc.py)
```python
class paddleseg.datasets.PascalVOC(transforms, dataset_root=None, mode='train', edge=False)
```
    PascalVOC2012 数据集 `http://host.robots.ox.ac.uk/pascal/VOC/`。
    如果你想对数据集做数据增强, 请运行 tools 中的 voc_augment.py。

### 参数
* **transforms** (list): 对图像的变换方法。
* **dataset_root** (str): 数据集存放目录。默认: None。
* **mode** (str, optional): 使用何种数据集。应为('train', 'val', 'test')中的一种。默认: 'train'。
        如果你想将 mode 设定为 'trainaug', 请确保该数据集已被增强过。 默认: 'train'。
* **edge** (bool, optional): 训练时是否指定求取边缘。默认: False。

## [ADE20K](../../../paddleseg/datasets/ade.py)
```python
class paddleseg.datasets.ADE20K(transforms, dataset_root=None, mode='train', edge=False)
```
    ADE20K 数据集 `http://sceneparsing.csail.mit.edu/`。

### 参数
* **transforms** (list): 由图像的变换方法构成的列表。
* **dataset_root** (str, optional): ADK20K 数据集存放目录。 默认: None。
* **mode** (str, optional): 全部数据集的一个子集， 应为('train', 'val')中的一种。 默认: 'train'。
* **edge** (bool, optional): 训练时是否指定求取边缘。默认: False。

## [OpticDiscSeg](../../../paddleseg/datasets/optic_disc_seg.py)
```python
class paddleseg.datasets.OpticDiscSeg(dataset_root=None, transforms=None, mode='train', edge=False)
```
    OpticDiscSeg 数据集取自 iChallenge-AMD `https://ai.baidu.com/broad/subordinate?dataset=amd`。

### 参数
* **transforms** (list): 对图像的变换方法。
* **dataset_root** (str): 数据集存放目录。默认: None。
* **mode** (str, optional): 使用何种数据集。应为('train', 'val', 'test')中的一种。默认: 'train'。
* **edge** (bool, optional): 训练时是否指定求取边缘。对Decouplednet训练时应指定该参数。默认: False。

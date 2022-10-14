简体中文|[English](marker.md)
# 准备自定义数据集

本文档首先介绍标注数据的背景知识，然后给出标注工具的教程，最后按照最常用的方式整理和切分数据集。

## 1、简介

### 1.1 标注图像格式

PaddleSeg使用单通道的标注图片，每一种像素值代表一种类别，像素标注类别需要从0开始递增，例如0，1，2，3表示有4种类别。

建议标注图像使用PNG无损压缩格式的图片，支持的标注类别最多为256类。

### 1.2 灰度标注图vs伪彩色标注图

通常标注图片是单通道的灰度图，显示是全黑效果，无法直接观察标注是否正确。
在原来的灰度标注图中注入调色板，就可以得到伪彩色的标注图。
灰度标注图和伪彩色标注图的对比如下。

PaddleSeg既支持灰度标注图，也支持伪彩色标注图。

<div align="center">
<img src="../image/image-11.png"  width = "600" />  
</div>

### 1.3 灰度标注图转换为伪彩色标注图 （非必须）

如果大家想将灰度标注图转换成伪彩色标注图，可使用PaddleSeg提供的转换工具。适用于以下两种常见的情况：

* 如果希望将指定目录下的所有灰度标注图转换为伪彩色标注图，则执行以下命令。

```buildoutcfg
python tools/data/gray2pseudo_color.py <dir_or_file> <output_dir>
```

|参数|用途|
|-|-|
|dir_or_file|灰度标注图的所在目录|
|output_dir|伪彩色标注图的保存目录|

* 如果仅希望将指定数据集中的部分灰度标注图转换为伪彩色标注图，则执行以下命令。

```buildoutcfg
python tools/data/gray2pseudo_color.py <dir_or_file> <output_dir> --dataset_dir <dataset directory> --file_separator <file list separator>
```
|参数|用途|
|-|-|
|dir_or_file|指定文件列表路径，该文件列表中的图片会进行转换|
|output_dir|彩色标注图的保存目录|
|--dataset_dir|数据集所在的根目录|
|--file_separator|文件列表的分隔符|

## 2、标注数据

如果不是使用已经标注好的公开数据集，大家需要预先采集图像，然后使用数据标注工具完成标注。

PddleSeg支持多种标注工具，比如EISeg交互式分割标注工具、LabelMe标注工具。标注工具的教程，请参考：
- [EISeg交互式分割标注工具教程](../../../EISeg/README.md)
- [LabelMe标注教程](../transform/transform_cn.md)

## 3、整理数据

标注完成，需要将数据整理成如下结构，即是所有原始图像存放在一个文件夹，所有标注图像存放在另一个文件夹。

注意，原始图像和标注图像的文件名是对应的（不要求后缀相同），请自行检查。

```
custom_dataset
    |
    |--images           # 存放所有原图
    |  |--image1.jpg
    |  |--image2.jpg
    |  |--...
    |
    |--labels           # 存放所有标注图
    |  |--label1.png
    |  |--label2.png
    |  |--...
```

## 4、切分数据

对于所有原始图像和标注图像，需要按照比例划分为训练集、验证集、测试集。

PaddleSeg提供了切分数据并生成文件列表的脚本。

```
python tools/data/split_dataset_list.py <dataset_root> <images_dir_name> <labels_dir_name> ${FLAGS}
```

参数说明：
- dataset_root: 数据集根目录
- images_dir_name: 原始图像目录名
- labels_dir_name: 标注图像目录名

FLAGS说明：

|FLAG|含义|默认值|参数数目|
|-|-|-|-|
|--split|训练集、验证集和测试集的切分比例|0.7 0.3 0|3|
|--separator|txt文件列表分隔符|" "|1|
|--format|原始图像和标注图像的图片后缀|"jpg"  "png"|2|
|--postfix|按文件主名（无扩展名）是否包含指定后缀对图片和标签集进行筛选|""   ""（2个空字符）|2|

使用示例：
```
python tools/data/split_dataset_list.py <dataset_root> images annotations --split 0.6 0.2 0.2 --format jpg png
```

运行后将在数据集根目录下生成`train.txt`、`val.txt`和`test.txt`，如下。

```
custom_dataset
    |
    |--images
    |  |--image1.jpg
    |  |--image2.jpg
    |  |--...
    |
    |--labels
    |  |--label1.png
    |  |--label2.png
    |  |--...
    |
    |--train.txt
    |
    |--val.txt
    |
    |--test.txt
```

三个txt文件的内容如下，每行是一张原始图片和标注图片的相对路径（相对于txt文件），两个相对路径中间是空格分隔符。
```
images/image1.jpg  annotations/image1.png
images/image2.jpg  annotations/image2.png
...
```

至此，自定义数据集准备完成。

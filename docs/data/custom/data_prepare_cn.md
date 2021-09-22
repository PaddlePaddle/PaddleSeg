简体中文|[English](data_prepare.md)
# 自定义数据集

## 1、如何使用数据集
我们希望将图像的路径写入到`train.txt`，`val.txt`，`test.txt`和`labels.txt`三个文件夹中，因为PaddleSeg是通过读取这些文本文件来定位图像路径的。
`train.txt`，`val.txt`和`test.txt`文本以空格为分割符分为两列，第一列为图像文件相对于dataset的相对路径，第二列为标注图像文件相对于dataset的相对路径。如下所示：
```
images/xxx1.jpg (xx1.png) annotations/xxx1.png
images/xxx2.jpg (xx2.png) annotations/xxx2.png
...
```
`labels.txt`: 每一行为一个单独的类别，相应的行号即为类别对应的id（行号从0开始)，如下所示：
```
labelA
labelB
...
```

## 2、切分自定义数据集

我们都知道，神经网络模型的训练过程通常要划分为训练集、验证集、测试集。如果你使用的是自定义数据集，PaddleSeg支持通过运行脚本的方式将数据集进行切分。如果你的数据集已经划分为以上三种，你可以跳过本步骤。

### 2.1 原图像要求
原图像数据的尺寸应为(h, w, channel)，其中h, w为图像的高和宽，channel为图像的通道数。

### 2.2 标注图要求
标注图像必须为单通道图像，标注图应为`png`格式。像素值即为对应的类别,像素标注类别需要从0开始递增。
例如0，1，2，3表示有4种类别，标注类别最多为256类。其中可以指定特定的像素值用于表示该值的像素不参与训练和评估（默认为255）。


### 2.3 自定义数据集切分与文件列表生成

对于未划分为训练集、验证集、测试集的全部数据，PaddleSeg提供了生成切分数据并生成文件列表的脚本。

#### 使用脚本对自定义数据集按比例随机切分，并生成文件列表
数据文件结构如下：
```
./dataset/  # 数据集根目录
|--images  # 原图目录
|  |--xxx1.jpg (xx1.png)
|  |--...
|  └--...
|
|--annotations  # 标注图目录
|  |--xxx1.png
|  |--...
|  └--...
```
其中，相应的文件名可根据需要自行定义。

使用命令如下，支持通过不同的Flags来开启特定功能。
```
python tools/split_dataset_list.py <dataset_root> <images_dir_name> <labels_dir_name> ${FLAGS}
```
参数说明：
- dataset_root: 数据集根目录
- images_dir_name: 原图目录名
- labels_dir_name: 标注图目录名

FLAGS说明：

|FLAG|含义|默认值|参数数目|
|-|-|-|-|
|--split|数据集切分比例|0.7 0.3 0|3|
|--separator|文件列表分隔符|" "|1|
|--format|图片和标签集的数据格式|"jpg"  "png"|2|
|--label_class|标注类别|'\_\_background\_\_' '\_\_foreground\_\_'|若干|
|--postfix|按文件主名（无扩展名）是否包含指定后缀对图片和标签集进行筛选|""   ""（2个空字符）|2|

运行后将在数据集根目录下生成`train.txt`，`val.txt`，`test.txt`和`labels.txt`.

**注:** 生成文件列表要求：要么原图和标注图片数量一致，要么只有原图，没有标注图片。若数据集缺少标注图片，将生成不含分隔符和标注图片路径的文件列表。

#### 使用示例
```
python tools/split_dataset_list.py <dataset_root> images annotations --split 0.6 0.2 0.2 --format jpg png
```



## 3.数据集文件整理

* 如果你需要使用自定义数据集进行训练，推荐整理成如下结构：
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

其中train.txt和val.txt的内容如下所示：

    images/image1.jpg labels/label1.png
    images/image2.jpg labels/label2.png
    ...

如果你只有划分好的数据集，可以通过执行以下脚本生成文件列表：
```
# 生成文件列表，其分隔符为空格，图片和标签集的数据格式都为png
python tools/create_dataset_list.py <your/dataset/dir> --separator " " --format png png
```
```
# 生成文件列表，其图片和标签集的文件夹名为img和gt，训练和验证集的文件夹名为training和validation，不生成测试集列表
python tools/create_dataset_list.py <your/dataset/dir> \
        --folder img gt --second_folder training validation
```
**注:** 必须指定自定义数据集目录，可以按需要设定FLAG。无需指定`--type`。
运行后将在数据集根目录下生成`train.txt`，`val.txt`，`test.txt`和`labels.txt`。PaddleSeg是通过读取这些文本文件来定位图像路径的。



* 标注图像的标签从0,1依次取值，不可间隔。若有需要忽略的像素，则按255进行标注。

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
请注意**数据集路径和训练文件**的存放位置，按照代码中的dataset_root和train_path示例方式存放。

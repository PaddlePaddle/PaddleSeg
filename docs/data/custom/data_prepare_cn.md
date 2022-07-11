简体中文|[English](data_prepare.md)
# 准备自定义数据集
如果您需要使用自定义数据集进行训练，请按照以下步骤准备数据。

## 数据集划分
我们建议将数据集划分为如下结构：

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
其中train.txt和val.txt的内容如下所示，用于PaddleSeg的[dataset父类](../../../paddleseg/datasets/dataset.py)读取数据：

```
    images/image1.jpg labels/label1.png
    images/image2.jpg labels/label2.png
    ...
```

### 自动化数据集划分
如果没有整理好数据，建议参考下列对文件进行划分并生成对应的txt。首先，将自定义数据集摆放成如下目录结构：
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

使用示例：
```
python tools/split_dataset_list.py <dataset_root> images annotations --split 0.6 0.2 0.2 --format jpg png
```


#### 生成文件列表

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


## 数据集配置
划分好的训练数据集可按如下方式对自定义数据集进行配置，评估数据集同理：
```yaml
train_dataset:
  type: Dataset
  dataset_root: the-relative-path-to-your-data
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

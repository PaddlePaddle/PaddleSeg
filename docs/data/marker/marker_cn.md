简体中文|[English](marker.md)
# 标注数据的准备

## 1、数据标注基础知识

### 1.1 标注协议
PaddleSeg采用单通道的标注图片，每一种像素值代表一种类别，像素标注类别需要从0开始递增，例如0，1，2，3表示有4种类别。

**注:** 标注图像请使用PNG无损压缩格式的图片。标注类别最多为256类。

### 1.2 灰度标注vs伪彩色标注
一般的分割库使用单通道灰度图作为标注图片，往往显示出来是全黑的效果。灰度标注图的弊端：
1. 对图像标注后，无法直接观察标注是否正确。
2. 模型测试过程无法直接判断分割的实际效果。

**PaddleSeg支持伪彩色图作为标注图片，在原来的单通道图片基础上，注入调色板。在基本不增加图片大小的基础上，却可以显示出彩色的效果。**

同时PaddleSeg也兼容灰度图标注，用户原来的灰度数据集可以不做修改，直接使用。
![](../image/image-11.png)

### 1.3 灰度标注转换为伪彩色标注
如果用户需要转换成伪彩色标注图，可使用我们的转换工具。适用于以下两种常见的情况：
1. 如果您希望将指定目录下的所有灰度标注图转换为伪彩色标注图，则执行以下命令，指定灰度标注所在的目录即可。
```buildoutcfg
python tools/gray2pseudo_color.py <dir_or_file> <output_dir>
```

|参数|用途|
|-|-|
|dir_or_file|指定灰度标注所在目录|
|output_dir|彩色标注图片的输出目录|

2. 如果您仅希望将指定数据集中的部分灰度标注图转换为伪彩色标注图，则执行以下命令，需要已有文件列表，按列表读取指定图片。
```buildoutcfg
python tools/gray2pseudo_color.py <dir_or_file> <output_dir> --dataset_dir <dataset directory> --file_separator <file list separator>
```
|参数|用途|
|-|-|
|dir_or_file|指定文件列表路径|
|output_dir|彩色标注图片的输出目录|
|--dataset_dir|数据集所在根目录|
|--file_separator|文件列表分隔符|


### 1.4 PaddleSeg如何使用数据集
我们希望将图像的路径写入到`train.txt`，`val.txt`，`test.txt`和`labels.txt`三个文件夹中，因为PaddleSeg是通过读取这些文本文件来定位图像路径的。
`train.txt`，`val.txt`和`test.txt`文本以空格为分割符分为两列，第一列为图像文件相对于dataset的相对路径，第二列为标注图像文件相对于dataset的相对路径。如下所示：
```
images/xxx1.jpg  annotations/xxx1.png
images/xxx2.jpg  annotations/xxx2.png
...
```
`labels.txt`: 每一行为一个单独的类别，相应的行号即为类别对应的id（行号从0开始)，如下所示：
```
labelA
labelB
...
```


## 2、标注自定义数据集
如果你想使用自定义数据集，你需预先采集好用于训练、评估和测试的图像，然后使用数据标注工具完成数据标注。若你想要使用Cityscapes、Pascal VOC等现成数据集，你可以跳过本步骤。

PddleSeg已支持2种标注工具：LabelMe、EISeg交互式分割标注工具。标注教程如下：

- [LabelMe标注教程](../transform/transform_cn.md)
- [EISeg交互式分割标注工具教程](../../../EISeg/README.md)

经以上工具进行标注后，请将所有的标注图像统一存放在annotations文件夹内，然后进行下一步。


## 3、切分自定义数据集


我们都知道，神经网络模型的训练过程通常要划分为训练集、验证集、测试集。如果你使用的是自定义数据集，PaddleSeg支持通过运行脚本的方式将数据集进行切分。若你想要使用Cityscapes、Pascal VOC等现成数据集，你可以跳过本步骤。

### 3.1 原图像要求
原图像数据的尺寸应为(h, w, channel)，其中h, w为图像的高和宽，channel为图像的通道数。

### 3.2 标注图要求
标注图像必须为单通道图像，像素值即为对应的类别,像素标注类别需要从0开始递增。
例如0，1，2，3表示有4种类别，标注类别最多为256类。其中可以指定特定的像素值用于表示该值的像素不参与训练和评估（默认为255）。


### 3.3 自定义数据集切分与文件列表生成

对于未划分为训练集、验证集、测试集的全部数据，PaddleSeg提供了生成切分数据并生成文件列表的脚本。
如果你的数据集已经像 Cityscapes、Pascal VOC等一样切分完成，请直接跳到第4节。否则，请参阅以下教程：


### 使用脚本对自定义数据集按比例随机切分，并生成文件列表
数据文件结构如下：
```
./dataset/  # 数据集根目录
|--images  # 原图目录
|  |--xxx1.jpg
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


## 4、数据集文件整理

PaddleSeg采用通用的文件列表方式组织训练集、验证集和测试集。在训练、评估、可视化过程前必须准备好相应的文件列表。

推荐整理成如下结构：

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

### 4.1 文件列表规范（训练、验证）

- 在训练与验证时，均需要提供标注图像。

- 即 `train.txt` 和 `val.txt` 的内容如下所示：
    ```
    images/image1.jpg labels/label1.png
    images/image2.jpg labels/label2.png
    ...
    ```

其中 `image1.jpg` 与 `label1.png` 分别为原始图像与其对应的标注图像。关于 `test.txt` 中的内容规范，请参照[4.2节](#4.2-文件列表规范（预测）)。


**注意事项**

* 务必保证分隔符在文件列表中每行只存在一次, 如文件名中存在空格，请使用"|"等文件名不可用字符进行切分。

* 文件列表请使用**UTF-8**格式保存, PaddleSeg默认使用UTF-8编码读取file_list文件。

* 需要保证文件列表的分割符与你的Dataset类保持一致，默认分割符为`空格`。


### 4.2 文件列表规范（预测）

- 在执行预测时，模型仅使用原始图像。

- 即 `test.txt` 的内容如下所示：
    ```
    images/image1.jpg
    images/image2.jpg
    ...
    ```

- 在调用`predict.py`进行可视化展示时，文件列表中可以包含标注图像。在预测时，模型将自动忽略文件列表中给出的标注图像。因此，你可以直接使用训练、验证数据集进行预测，而不必修改 [4.1节](#4.1-文件列表规范（训练、验证）)里 `train.txt` 和 `val.txt` 文件中的内容。



### 4.3 数据集目录结构整理

如果用户想要生成数据集的文件列表，需要整理成如下的目录结构（类似于Cityscapes数据集）。你可以手动划分，亦可参照第3节中使用脚本自动切分生成的方式。

```
./dataset/   # 数据集根目录
├── annotations      # 标注图像目录
│   ├── test
│   │   ├── ...
│   │   └── ...
│   ├── train
│   │   ├── ...
│   │   └── ...
│   └── val
│       ├── ...
│       └── ...
└── images       # 原图像目录
    ├── test
    │   ├── ...
    │   └── ...
    ├── train
    │   ├── ...
    │   └── ...
    └── val
        ├── ...
        └── ...
注：以上目录名可任意
```

### 4.4 生成文件列表
PaddleSeg提供了生成文件列表的使用脚本，可适用于自定义数据集或cityscapes数据集，并支持通过不同的Flags来开启特定功能。
```
python tools/create_dataset_list.py <your/dataset/dir> ${FLAGS}
```
运行后将在数据集根目录下生成训练/验证/测试集的文件列表（文件主名与`--second_folder`一致，扩展名为`.txt`）。

**注:** 生成文件列表要求：要么原图和标注图片数量一致，要么只有原图，没有标注图片。若数据集缺少标注图片，仍可自动生成不含分隔符和标注图片路径的文件列表。

#### 命令行FLAGS列表

|FLAG|用途|默认值|参数数目|
|-|-|-|-|
|--type|指定数据集类型，`cityscapes`或`自定义`|`自定义`|1|
|--separator|文件列表分隔符|"&#124;"|1|
|--folder|图片和标签集的文件夹名|"images" "annotations"|2|
|--second_folder|训练/验证/测试集的文件夹名|"train" "val" "test"|若干|
|--format|图片和标签集的数据格式|"jpg"  "png"|2|
|--postfix|按文件主名（无扩展名）是否包含指定后缀对图片和标签集进行筛选|""   ""（2个空字符）|2|

#### 使用示例
- **对于自定义数据集**

若您已经按上述说明整理好了数据集目录结构，可以运行下面的命令生成文件列表。

```
# 生成文件列表，其分隔符为空格，图片和标签集的数据格式都为png
python tools/create_dataset_list.py <your/dataset/dir> --separator " " --format jpg png
```
```
# 生成文件列表，其图片和标签集的文件夹名为img和gt，训练和验证集的文件夹名为training和validation，不生成测试集列表
python tools/create_dataset_list.py <your/dataset/dir> \
        --folder img gt --second_folder training validation
```
**注:** 必须指定自定义数据集目录，可以按需要设定FLAG。无需指定`--type`。

- **对于cityscapes数据集**

若您使用的是cityscapes数据集，可以运行下面的命令生成文件列表。

```
# 生成cityscapes文件列表，其分隔符为逗号
python tools/create_dataset_list.py <your/dataset/dir> --type cityscapes --separator ","
```
**注:**

必须指定cityscapes数据集目录，`--type`必须为`cityscapes`。

在cityscapes类型下，部分FLAG将被重新设定，无需手动指定，具体如下：

|FLAG|固定值|
|-|-|
|--folder|"leftImg8bit" "gtFine"|
|--format|"jpg" "png"|
|--postfix|"_leftImg8bit" "_gtFine_labelTrainIds"|

其余FLAG可以按需要设定。



运行后将在数据集根目录下生成`train.txt`，`val.txt`，`test.txt`和`labels.txt`。PaddleSeg是通过读取这些文本文件来定位图像路径的。

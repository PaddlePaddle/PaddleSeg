# 数据准备


## 数据协议
数据集包含原图、标注图及相应的文件列表文件。

### 数据格式
遥感影像的格式多种多样，不同传感器产生的数据格式也可能不同。  
PaddleSeg已兼容以下4种格式图片读取：
- `tif`
- `png`
- `img`
- `npy`

### 原图要求
原图数据的尺寸应为(h, w, channel)，其中h, w为图像的高和宽，channel为图像的通道数。

### 标注图要求
标注图像必须为单通道图像，像素值即为对应的类别,像素标注类别需要从0开始递增。
例如0，1，2，3表示有4种类别，标注类别最多为256类。其中可以指定特定的像素值用于表示该值的像素不参与训练和评估（默认为255）。

### 文件列表文件
文件列表文件包括`train.txt`，`val.txt`，`test.txt`和`labels.txt`.  
`train.txt`，`val.txt`和`test.txt`文本以空格为分割符分为两列，第一列为图像文件相对于dataset的相对路径，第二列为标注图像文件相对于dataset的相对路径。如下所示：
```
images/xxx1.tif annotations/xxx1.png
images/xxx2.tif annotations/xxx2.png
...
```
`labels.txt`: 每一行为一个单独的类别，相应的行号即为类别对应的id（行号从0开始)，如下所示：
```
labelA
labelB
...
```

## 数据集切分和文件列表生成
数据集切分有2种方式：随机切分和手动切分。对于这2种方式，PaddleSeg均提供了生成文件列表的脚本，您可以按需要选择。

### 1 对数据集按比例随机切分，并生成文件列表
数据文件结构如下：
```
./dataset/  # 数据集根目录
|--images  # 原图目录
|  |--xxx1.tif
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
|--format|图片和标签集的数据格式|"tif"  "png"|2|
|--label_class|标注类别|'\_\_background\_\_' '\_\_foreground\_\_'|若干|
|--postfix|按文件主名（无扩展名）是否包含指定后缀对图片和标签集进行筛选|""   ""（2个空字符）|2|

运行后将在数据集根目录下生成`train.txt`，`val.txt`，`test.txt`和`labels.txt`.

**Note:** 生成文件列表要求：要么原图和标注图片数量一致，要么只有原图，没有标注图片。若数据集缺少标注图片，将生成不含分隔符和标注图片路径的文件列表。

#### 使用示例
```
python tools/split_dataset_list.py <dataset_root> images annotations --split 0.6 0.2 0.2 --format tif png
```

### 2 已经手工划分好数据集，按照目录结构生成文件列表
数据目录手工划分成如下结构：
```
./dataset/   # 数据集根目录
├── annotations      # 标注目录
│   ├── test
│   │   ├── ...
│   │   └── ...
│   ├── train
│   │   ├── ...
│   │   └── ...
│   └── val
│       ├── ...
│       └── ...
└── images       # 原图目录
    ├── test
    │   ├── ...
    │   └── ...
    ├── train
    │   ├── ...
    │   └── ...
    └── val
        ├── ...
        └── ...
```
其中，相应的文件名可根据需要自行定义。

使用命令如下，支持通过不同的Flags来开启特定功能。
```
python tools/create_dataset_list.py <dataset_root> ${FLAGS}
```
参数说明：
- dataset_root: 数据集根目录

FLAGS说明：

|FLAG|含义|默认值|参数数目|
|-|-|-|-|
|--separator|文件列表分隔符|" "|1|
|--folder|图片和标签集的文件夹名|"images" "annotations"|2|
|--second_folder|训练/验证/测试集的文件夹名|"train" "val" "test"|若干|
|--format|图片和标签集的数据格式|"tif"  "png"|2|
|--label_class|标注类别|'\_\_background\_\_' '\_\_foreground\_\_'|若干|
|--postfix|按文件主名（无扩展名）是否包含指定后缀对图片和标签集进行筛选|""   ""（2个空字符）|2|

运行后将在数据集根目录下生成`train.txt`，`val.txt`，`test.txt`和`labels.txt`.

**Note:** 生成文件列表要求：要么原图和标注图片数量一致，要么只有原图，没有标注图片。若数据集缺少标注图片，将生成不含分隔符和标注图片路径的文件列表。

#### 使用示例
若您已经按上述说明整理好了数据集目录结构，可以运行下面的命令生成文件列表。

```
# 生成文件列表，其分隔符为空格，图片和标签集的数据格式都为png
python tools/create_dataset_list.py <dataset_root> --separator " " --format png png
```
```
# 生成文件列表，其图片和标签集的文件夹名为img和gt，训练和验证集的文件夹名为training和validation，不生成测试集列表
python tools/create_dataset_list.py <dataset_root> \
        --folder img gt --second_folder training validation
```

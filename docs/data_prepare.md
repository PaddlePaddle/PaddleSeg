# PaddleSeg 数据准备

## 数据标注

数据标注推荐使用LabelMe工具，具体可参考文档[PaddleSeg 数据标注](./annotation/README.md)


## 语义分割标注规范

PaddleSeg采用通用的文件列表方式组织训练集、验证集和测试集。像素标注类别需要从0开始递增。

**NOTE:** 标注图像请使用PNG无损压缩格式的图片

以Cityscapes数据集为例, 我们需要整理出训练集、验证集、测试集对应的原图和标注文件列表用于PaddleSeg训练即可。

其中`DATASET.DATA_DIR`为数据根目录，文件列表的路径以数据集根目录作为相对路径起始点。

```
./cityscapes/   # 数据集根目录
├── gtFine      # 标注目录
│   ├── test
│   │   ├── berlin
│   │   └── ...
│   ├── train
│   │   ├── aachen
│   │   └── ...
│   └── val
│       ├── frankfurt
│       └── ...
└── leftImg8bit  # 原图目录
    ├── test
    │   ├── berlin
    │   └── ...
    ├── train
    │   ├── aachen
    │   └── ...
    └── val
        ├── frankfurt
        └── ...
```

文件列表组织形式如下
```
原始图片路径 [SEP] 标注图片路径
```


其中`[SEP]`是文件路径分割符，可以在`DATASET.SEPARATOR`配置项中修改, 默认为空格。

**注意事项**

* 务必保证分隔符在文件列表中每行只存在一次, 如文件名中存在空格，请使用'|'等文件名不可用字符进行切分

* 文件列表请使用**UTF-8**格式保存, PaddleSeg默认使用UTF-8编码读取file_list文件

如下图所示，左边为原图的图片路径，右边为图片对应的标注路径。

![cityscapes_filelist](./imgs/file_list.png)

完整的配置信息可以参考[`./dataset/cityscapes_demo`](../dataset/cityscapes_demo/)目录下的yaml和文件列表。

## 数据校验
从7方面对用户自定义的数据集和yaml配置进行校验，帮助用户排查基本的数据和配置问题。

数据校验脚本如下，支持通过`YAML_FILE_PATH`来指定配置文件。
```
# YAML_FILE_PATH为yaml配置文件路径
python pdseg/check.py --cfg ${YAML_FILE_PATH}
```
### 1 数据集基本校验
* 数据集路径检查，包括`DATASET.TRAIN_FILE_LIST`，`DATASET.VAL_FILE_LIST`，`DATASET.TEST_FILE_LIST`设置是否正确。
* 列表分割符检查，判断在`TRAIN_FILE_LIST`，`VAL_FILE_LIST`和`TEST_FILE_LIST`列表文件中的分隔符`DATASET.SEPARATOR`设置是否正确。

### 2 标注类别校验
检查实际标注类别是否和配置参数`DATASET.NUM_CLASSES`，`DATASET.IGNORE_INDEX`匹配。

**NOTE:**
标注图像类别数值必须在[0~(`DATASET.NUM_CLASSES`-1)]范围内或者为`DATASET.IGNORE_INDEX`。
标注类别最好从0开始，否则可能影响精度。

### 3 标注像素统计
统计每种类别像素数量，显示以供参考。

### 4 标注格式校验
检查标注图像是否为PNG格式。

**NOTE:** 标注图像请使用PNG无损压缩格式的图片，若使用其他格式则可能影响精度。

### 5 图像格式校验
检查图片类型`DATASET.IMAGE_TYPE`是否设置正确。

**NOTE:** 当数据集包含三通道图片时`DATASET.IMAGE_TYPE`设置为rgb；
当数据集全部为四通道图片时`DATASET.IMAGE_TYPE`设置为rgba；

### 6 图像与标注图尺寸一致性校验
验证图像尺寸和对应标注图尺寸是否一致。

### 7 模型验证参数`EVAL_CROP_SIZE`校验
验证`EVAL_CROP_SIZE`是否设置正确，共有3种情形：

- 当`AUG.AUG_METHOD`为unpadding时，`EVAL_CROP_SIZE`的宽高应不小于`AUG.FIX_RESIZE_SIZE`的宽高。

- 当`AUG.AUG_METHOD`为stepscaling时，`EVAL_CROP_SIZE`的宽高应不小于原图中最大的宽高。

- 当`AUG.AUG_METHOD`为rangscaling时，`EVAL_CROP_SIZE`的宽高应不小于缩放后图像中最大的宽高。

我们将计算并给出`EVAL_CROP_SIZE`的建议值。

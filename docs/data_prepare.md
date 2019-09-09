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

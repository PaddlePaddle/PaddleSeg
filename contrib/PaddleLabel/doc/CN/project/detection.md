# 目标检测标注

![image](https://user-images.githubusercontent.com/29757093/182841361-eb53e726-fa98-4e02-88ba-30172efac8eb.png)

PaddleLabel 支持图像目标检测标注任务。

## <div id="dataset_structure">数据结构</div>

PaddleLabel 目前支持 PASCAL VOC，COCO 和 YOLO 三种目标检测数据集格式。

### PASCAL VOC

PASCAL VOC 格式将标注信息保存在 xml 文件中，每张图像对应一个 xml 文件。新建标注项目时，填写的`数据集路径`下所有图片都将被导入，标签和图像对应规则在下一段详述。

示例格式如下：

```shell
数据集路径
├── Annotations
│   ├── 0001.xml
│   ├── 0002.xml
│   ├── 0003.xml
│   └── ...
├── JPEGImages
│   ├── 0001.jpg
│   ├── 0002.jpg
│   ├── 0003.jpg
│   └── ...
├── labels.txt
├── test_list.txt
├── train_list.txt
└── val_list.txt
```

xml 文件格式如下：

```text
<annotation>
 <folder>JPEGImages</folder>
 <filename></filename>
 <source>
  <database></database>
 </source>
 <size>
  <width></width>
  <height></height>
  <depth></depth>
 </size>
 <segmented>0</segmented>
 <object>
  <name></name>
  <pose></pose>
  <truncated></truncated>
  <difficult></difficult>
  <bndbox>
   <xmin></xmin>
   <ymin></ymin>
   <xmax></xmax>
   <ymax></ymax>
  </bndbox>
 </object>
</annotation>
```

导入 VOC 格式数据集时，PaddleLabel 将把数据集路径下所有 .xml 结尾文件作为标签导入，并将该标签与位于`/数据集路径/folder/filename`的图像文件匹配。图像路径中的`folder`和`filename`将从该 xml 文件中解析。如果 xml 中没有`folder`节点，将使用默认值 JPEGImages。如果`folder`节点内容为空，将认为图像文件位于`/数据集路径/filename`。xml 中 filename 节点必须存在，否则导入会失败。如果导入图像后发现有 xml 标注信息的图像中没有标注，可以查看 PaddleLabel 运行的命令行

### COCO

COCO 格式将整个数据集的所有标注信息存在一个（或少数几个）`json`文件中。这里列出了 COCO 和检测相关的部分格式规范，更多细节请访问[COCO 官网](https://cocodataset.org/#format-data)。下文没有列出的项不会在导入时被保存到数据库中和最终导出，比如图像的 date_captured 属性。 `注意，所有使用 COCO 格式的项目都不支持[xx_list.txt](./common.md#xxlisttxt)和[labels.txt](./common.md#labelstxt)。`新建标注项目时，填写的`数据集路径`下所有图片都将被导入，标签和图像对应规则在下一段详述。

示例格式如下：

```shell
数据集路径
├── image
│   ├── 0001.jpg
│   ├── 0002.jpg
│   ├── 0003.jpg
│   └── ...
├── train.json
├── val.json
└── test.json
```

COCO 文件的格式如下：

```text
{
    "info": info,
    "images": [image],
    "annotations": [annotation],
    "licenses": [license],
    "categories": [category],
}

image{
    "id": int,
    "width": int,
    "height": int,
    "file_name": str,
}

annotation{
    "id": int,
    "image_id": int,
    "category_id": int,
    "area": float,
    "bbox": [x, y, width, height],
}

category{
 "id": int,
 "name": str,
 "supercategory": str,
 "color": str // PaddleLabel加入功能，COCO官方定义中没有这一项。color会被导出，导入时如果存在PaddleLabel会给这一类别赋color指定的颜色
}
```

PaddleLabel将coco标注信息中的图片记录和盘上的图片对应起来的逻辑为：image\['file_name'\]中最后的文件名和盘上图片的文件名相同（大小写敏感）。这个设计是为了让对应逻辑尽可能简单并保持一定的跨平台兼容性。推荐将所有图片放在同一个文件夹下以避免图片重名导致coco标注信息中的一个图片记录对应到盘上的多张图片。一些标注工具导出的coco标注记录中，image\['file_name'\]项可能是完整的文件路径或相对数据集根目录的路径，这种情况下我们用'/'和''分割这个路径，得到其中的文件名。因此请避免在文件名中使用'/'和''。

### YOLO

YOLO 格式每张图像对应一个 txt 格式的标注信息文件，二者文件名除拓展名部分相同。

示例格式如下：

```shell
数据集路径
├── Annotations
│   ├── 0001.txt
│   ├── 0002.txt
│   ├── 0003.txt
│   └── ...
├── JPEGImages
│   ├── 0001.jpg
│   ├── 0002.jpg
│   ├── 0003.jpg
│   └── ...
├── labels.txt
├── test_list.txt
├── train_list.txt
└── val_list.txt
```

```txt
# 0001.txt
0 0.4 0.5 0.7 0.8
# 标签id bb中心宽方向位置/图像宽 bb中心长方向位置/图像长 bb宽/图像宽 bb长/图像长
# 标签id从0开始
# 导入时没有标注的图像可以不提供标注文件，或提供空文件。PaddleLabel对没有标注的图像不会导出空YOLO标注文件
```

注意 YOLO 格式的图像和标签完全通过文件名对应，如果有两张图片文件名只有拓展名不同，比如 cat.png 和 cat.jpeg，二者都会和 cat.txt 标签文件对应。为了避免这一情况，PaddleLabel 在导入图像时遇到上述情况会将 cat.png 重命名为 cat-1.png。这可能会导致图像找不到对应标注文件。如果发现有图像提供了标注文件但是导入后没有标注，可以查看 PaddleLabel 运行的命令行并对文件名做出调整后重新导入项目。

此外建议将所有图像都放在同一文件夹下，避免重名导致图像标注对应问题。

### 创建新项目

浏览器打开 PaddleLabel 后，可以通过创建项目下的目标检测卡片创建一个新的目标检测标注项目（如果已经创建，可以通过下方“我的项目”找到对应项目，点击“标注”继续标注）。

项目创建选项卡有如下选项需要填写：

- 项目名称（必填）：填写该检测标注项目的项目名
- 数据集路径（必填）：填写本地数据集的根文件夹，参考上方实力格式。建议从文件管理器复制粘贴
- 数据集描述（选填）：该标注项目的简单描述
- 标签格式（选填）：导入数据集的格式，如果没有标注信息不需要选择

### 数据导入

在创建项目时需要填写数据地址，该地址对应的是数据集的文件夹，为了使 PaddleLabel 能够正确的识别和处理数据集，请参考[数据结构](#dataset_structure)组织待标注文件结构，具体 txt 文件的说明可参考[数据集文件结构说明](dataset_file_structure.md)。同时 PaddleLabel 提供了参考数据集，位于`~/.paddlelabel/sample/detection` 文件夹下，也可参考该数据集文件结构组织数据。

## 数据标注

完成后进入标注界面，PaddleLabel 的界面分为五个区域，上方为可以切换的标签页，下方为标注进度展示，左侧包含图像显示区域与工具栏，右侧为标签列表，用于添加不同的标签和标注。在检测任务的标注中，可以按以下步骤进行使用：

1. 点击右侧“添加标签”，填写信息并创建标签
1. 选择一个标签，点击左侧工具栏的“矩形”，在图像界面上点击需要标注的物体的左上角和右下角，当出现矩形框时完成当前目标的标注。通过点击"编辑"，可修改矩形框的大小和位置。
1. 点击左右按钮切换图像，重复上述操作，直到所有数据标注完毕
1. 下方进度展示可以查看标注进度

_注意：在 PaddleLabel 中，右侧标签栏有标签列表和标注列表两种。在目标检测中，标签列表对应的是类别，而标注列表对应的是该类别的一个实例。_

## 完成标注

完成数据标注后，PaddleLabel 提供了方便的数据划分功能，以便与 Paddle 其他工具套件（如 PaddleDetection）进行快速衔接。点击右侧工具栏的“项目总览”按钮，来到该项目的总览界面，这里可以看到数据以及标注状态。

### 数据划分

点击**划分数据集**按钮弹出划分比例的设置，分别填入对应训练集、验证集和测试集的占比，点击确定即可完成数据集的划分。

### 数据导出

点击**导出数据集**，输入需要导出到的文件夹路径，点击确认，即可导出标注完成的数据到指定路径。

## \*检测预标注

PaddleLabel带有基于PaddlePaddle的机器学习检测标注功能，可以通过加载模型实现检测预标注功能，使用方法参考[图像检测自动标注](detection_auto_label.md)。

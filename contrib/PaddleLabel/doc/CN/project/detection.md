# 目标检测标注

![image](https://user-images.githubusercontent.com/29757093/182841361-eb53e726-fa98-4e02-88ba-30172efac8eb.png)

PaddleLabel 支持图像目标检测标注任务。

## <div id="test2">数据结构</div>

PaddleLabel 支持 PASCAL VOC 和 COCO 两种目标检测数据集格式。

### PASCAL VOC

PASCAL VOC 格式将标注信息保存在 xml 文件中，每个图像都对应一个 xml 文件。新建标注任务时，待标注的图片放于`JPEGImages`文件夹下，数据集路径填写`JPEGImages`上层目录`Dataset Path`即可。
示例格式如下：

```shell
Dataset Path
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

导入 VOC 格式数据集时，PaddleLabel 将把数据集路径下所有 xml 结尾文件作为标签，并将该标签与位于`/数据集路径/folder/filename`的图像文件匹配。路径中的`folder`和`filename`将从该 xml 文件中解析。如果 xml 中没有`folder`节点，默认值是 JPEGImages。如果`folder`节点内容为空，将认为图像文件位于`/数据集路径/filename`。

### COCO

COCO 格式将整个数据集的所有标注信息存在一个`json`文件中。这里列出了 COCO 的部分格式规范，更多细节请访问[COCO 官网](https://cocodataset.org/#format-data)。注意，所有使用 COCO 格式的项目都不支持`xx_list.txt`和`labels.txt`。新建标注任务时，待标注的图片放于`image`文件夹下，数据集路径填写`JPEGImages`上层目录`Dataset Path`即可。

示例格式如下：

```shell
Dataset Path
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
    "license": int,
    "flickr_url": str,
    "coco_url": str,
    "date_captured": datetime,
}

annotation{
    "id": int,
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [polygon],
    "area": float,
    "bbox": [x,y,width,height],
    "iscrowd": 0 or 1,
}

category{
	"id": int,
	"name": str,
	"supercategory": str,
	"color": str // this feature is specific to PP Label. It's not in the coco spec.
}
```

### 创建新项目

浏览器打开 PaddleLabel 后，可以通过创建项目下的目标检测”卡片创建一个新的目标检测标注项目（如果已经创建，可以通过下方“我的项目”找到对应名称的项目，点击“标注”继续标注）。

项目创建选项卡有如下选项需要填写：

- 项目名称（必填）：填写该检测标注项目的项目名
- 数据地址（必填）：填写本地数据集文件夹的路径，可以直接通过复制路径并粘贴得到。
- 数据集描述（选填）：填写该检测标注项目的使用的数据集的描述文字
- 标签格式（必选）：选择该任务的数据集格式为 COCO 还是 VOC

### 数据导入

在创建项目时需要填写数据地址，该地址对应的是数据集的文件夹，为了使 PaddleLabel 能够正确的识别和处理数据集，请参考[数据结构](#test2)组织待标注文件结构，具体 txt 文件的说明可参考[数据集文件结构说明](dataset_file_structure.md)。同时 PaddleLabel 提供了参考数据集，位于`~/.paddlelabel/sample/det`路径下，也可参考该数据集文件结构组织数据。

## 数据标注

完成后进入标注界面，PaddleLabel 的界面分为五个区域，上方为可以切换的标签页，下方为标注进度展示，左侧包含图像显示区域与工具栏，右侧为标签列表，用于添加不同的标签和标注。在检测任务的标注中，可以按以下步骤进行使用：

1. 点击右侧“添加标签”，填写信息并创建标签
2. 选择一个标签，点击左侧工具栏的“矩形”，在图像界面上点击需要标注的物体的左上角和右下角，当出现矩形框时完成当前目标的标注。通过点击"编辑"，可修改矩形框的大小和位置。
3. 点击左右按钮切换图像，重复上述操作，直到所有数据标注完毕
4. 下方进度展示可以查看标注进度

_注意：在 PaddleLabel 中，右侧标签栏有标签列表和标注列表两种。在目标检测中，标签列表对应的是类别，而标注列表对应的是该类别的一个实例。_

## 完成标注

完成数据标注后，PaddleLabel 提供了方便的数据划分功能，以便与 Paddle 其他工具套件（如 PaddleDetection）进行快速衔接。点击右侧工具栏的“项目总览”按钮，来到该项目的总览界面，这里可以看到数据以及标注状态。

### 数据划分

点击**划分数据集**按钮弹出划分比例的设置，分别填入对应训练集、验证集和测试集的占比，点击确定即可完成数据集的划分。

### 数据导出

点击**导出数据集**，输入需要导出到的文件夹路径，点击确认，即可导出标注完成的数据到指定路径。

# 数据集文件结构说明

本页面介绍 PaddleLabel 可以导入/导出的数据集文件结构，以帮助您更好的使用 PaddleLabel。**首先需要注意，PaddleLabel 可能修改数据集文件夹下的文件**。比如在`导入更多数据`时，新的数据文件将被移动到这个项目的数据集文件夹中。这一设计的目的在于避免复制数据集以节省磁盘空间。目前 PaddleLabel 不会删除盘上的任何内容，**但建议您在导入之前复制数据集作为备份**。使用中 PaddleLabel 会在该数据集根文件夹下创建一个名为`paddlelabel.warning`的文件。请避免更改文件夹下的任何文件以防出现问题。

PaddleLabel 为每一种支持的标注项目都内置了样例数据集。可以通过点击欢迎页面的“样例项目”按钮，选择任务类型进行创建。创建一个样例项目后，所有样例项目数据将被解压到`~/.paddlelabel/sample`文件夹中，可以作为参考。

## 无标注数据集

如果您的数据集不包含任何标注，只需将所有图像文件放在一个文件夹下。 PaddleLabel 会遍历文件夹（及所有子文件夹）中所有文件，并按照**文件拓展名**判断其类型，导入所有图像文件。所有隐藏文件（文件名以`.`开头）将被忽略。

## 基础功能

不同类型项目的数据集文件结构有所不同，但大多数类型的项目都支持一些基础功能。

### labels.txt

所有不使用 COCO 格式保存标注的项目都支持`labels.txt`。PaddleLabel 在导入过程中会在数据集路径下寻找`labels.txt`文件。您可以在这个文件中列出该项目的所有标签（每行一个）。例如下面这样:

```text
# labels.txt
Monkey
Mouse
```

PaddleLabel 的标签名称支持任何字符串，但是标签名称可能被用作导出数据集的文件夹名，所以应避免任何您的操作系统不支持的字符串，可以参考[这篇回答](https://stackoverflow.com/a/31976060)。Paddle 生态中的其他工具对标签名可能有进一步限制，如[PaddleX](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/data/format/classification.md)不支持中文字符作为标签名称。

在导入过程中，`labels.txt`可以包含标签名以外的信息。目前支持 4 种格式，如下所示。其中`|`表示分隔符，默认为空格。

标签长度：

- 1：标签名
- 2：标签名 | 标签编号
- 3：标签名 | 标签编号 | 十六进制颜色或常用颜色名称或灰度值
- 5：标签名 | 标签编号 | 红色 | 绿色 | 蓝色

其他：

- `//`：`//`后的字符串将被作为标签注释
- `-`：如果需要指定标签颜色，但不想指定标签编号，在标签编号位置写`-`

一些例子：

```text
dog
monkey 4
mouse - #0000ff // mouse's id will be 5
cat 10 yellow
zibra 11 blue // some common colors are supported
snake 12 255 0 0 // rgb color
```

所有支持的颜色名称在[这里](https://github.com/PaddleCV-SIG/PaddleLabel/blob/develop/paddlelabel/task/util/color.py#L15)列出。

在导入过程中，PaddleLabel 会首先创建`labels.txt`中指定的标签。因此这个文件中的标签的编号将从**0**开始并递增。在导出过程中也将生成此文件。

### xx_list.txt

所有不使用 COCO 格式保存标注的项目都支持`xx_list.txt`。`xx_list.txt `包括`train_list.txt`，` val_list.txt`和`test_list.txt`。这三个文件需要放在数据集文件夹的根目录中，与`labels.txt`相同。

这三个文件指定了数据集的划分以及标签或标注文件与图像文件间的匹配关系（比如 voc 格式下，每一行是图像文件的路径和标签文件的路径）。这三个文件的内容结构相同，每一行都以一条数据的路径开始，其路径为相对数据集根目录的相对路径，后面跟着表示类别的整数/字符串，或者标签文件的路径。例如：

```text
# train_list.txt
image/9911.jpg 0 3
image/9932.jpg 4
image/9928.jpg Cat
```

或

```text
# train_list.txt
JPEGImages/1.jpeg Annotations/1.xml
JPEGImages/2.jpeg Annotations/2.xml
JPEGImages/3.jpeg Annotations/3.xml
```

需要注意的是，**大多数项目都只会用到`xx_list.txt`中的数据集划分信息**。

如果标签类别为整数，PaddleLabel 将在`labels.txt`中查找标签，标签 id 从**0**开始。一些数据集中一条数据可以有多个类别，比如图像多分类。如果希望用数字作为标签名称，您可以将数字写在`labels.txt`中，并在`xx_list.txt`中提供标签 id。或者可以给数字标签加一个前缀，例如将`10`表示为`n10`。

这三个文件都将在导出过程中生成，即使其中一些文件是空的。注意，为了确保这些文件可以被 Paddle 生态系统中的其他工具读取，没有注释的数据**不会包含在`xx_list.txt`中**。

## 图像分类

PaddleLabel 支持单分类和多分类。

### 单分类

也称为 ImageNet 格式。样例数据集：[flowers102](https://paddle-imagenet-models-name.bj.bcebos.com/data/flowers102.zip)、[vegetables_cls](https://bj.bcebos.com/paddlex/datasets/vegetables_cls.tar.gz)。

示例格式如下：

```shell
Dataset Path
├── Cat
│   ├── cat-1.jpg
│   ├── cat-2.png
│   ├── cat-3.webp
│   └── ...
├── Dog
│   ├── dog-1.jpg
│   ├── dog-2.jpg
│   ├── dog-3.jpg
│   └── ...
├── monkey.jpg
├── train_list.txt
├── val_list.txt
├── test_list.txt
└── label.txt

# labels.txt
Monkey
Mouse
```

单分类中图像所在的文件夹名称将被视为它的类别。所以如上数据集导入后，三张猫和三张狗的图片会有分类，monkey.jpg 没有分类。如果与文件夹名同名的标签不存在，导入过程中会自动创建。

为了避免冲突，PaddleLabel 只使用`xx_list.txt`中的数据集划分信息，**这三个文件中的类别信息将不会被考虑**。您可以使用[此脚本](../tool/clas/mv_image_acc_split.py)在导入数据之前根据三个`xx_list.txt`文件更改数据的位置。

### 多分类

在多分类项目中，一条数据可以有多个类别。

示例格式如下：

```shell
Dataset Path
├── image
│   ├── 9911.jpg
│   ├── 9932.jpg
│   └── monkey.jpg
├── labels.txt
├── test_list.txt
├── train_list.txt
└── val_list.txt

# labels.txt
cat
dog
yellow
black

# train_list.txt
image/9911.jpg 0 3
image/9932.jpg 4 0
image/9928.jpg monkey
```

在多分类项目中，数据的类别仅由`xx_list.txt`决定，不会考虑文件夹名称。

## 目标检测

PaddleLabel 支持 PASCAL VOC 和 COCO 两种目标检测数据集格式。

### PASCAL VOC

PASCAL VOC 格式将标注信息保存在 xml 文件中，每个图像都对应一个 xml 文件。样例数据集：[昆虫检测数据集](https://bj.bcebos.com/paddlex/datasets/insect_det.tar.gz)。

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

COCO 格式将整个数据集的所有标注信息存在一个`json`文件中。这里列出了 COCO 的部分格式规范，更多细节请访问[COCO 官网](https://cocodataset.org/#format-data)。注意，所有使用 COCO 格式的项目都不支持`xx_list.txt`和`labels.txt`。样例数据集：[Plane Detection]()。

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

#### 数据导入

PaddleLabel 使用[pycocotoolse](https://github.com/linhandev/cocoapie)解析标注文件。pycocotoolse 与原版 [pycocotools](https://github.com/cocodataset/cocoapi)基本相同，仅在其基础上增加了一些数据集管理功能。导入过程中 PaddleLabel 会在数据集路径下寻找三个 json 文件：`train.json`、`val.json`和`test.json`，并从这三个文件中解析出用于训练、验证和测试的数据。请确保**每个图像在三个 json 中只被定义一次**，否则将导入失败。

在导入时PaddleLabel 会导入数据集文件夹下的所有图像。COCO json 中每张图有一个 `file_name`，对应匹配一张图的路径。因此如果发现一张图片有多条匹配的记录，导入会失败。例如路径为`\Dataset Path\folder1\image.png`和`\Dataset Path\folder2\image.png`的两张图像都将与`file_name`为“image.png”的图像匹配。**建议将所有图像放在一个文件夹下，以避免图像重名**。

如果一个图像的记录中没有包含宽度或高度的信息，PaddleLabel 将在导入期间读取图像来获取。这将拖慢数据集导入速度。

#### 数据导出

导出过程中，即使将所有数据都划分为同一子集（如训练集），上述三个 COCO json 文件也都会生成。

在 COCO json 的分类部分，PaddleLabel 添加了一个颜色字段。这个字段不在标准的 COCO 结构中。颜色字段会导出保存，并在导入时使用。

## 图像分割

PaddleLabel 支持两种类型的分割任务（语义分割和实例分割）和两种数据集格式（掩膜格式和多边形格式）。语义分割和实例分割中，多边形格式中是完全相同的，二者保存掩膜格式存在区别。

### 多边形格式

PaddleLabel 使用 COCO 格式将分割结果存为多边形。其导入/导出过程与[目标检测项目中使用 COCO 格式](#coco)的过程基本相同。

### 掩膜格式

进行语义分割时，只需要确定输入图像中每个像素属于哪一类。输出是和输入图像大小相同的 png，每个像素的灰度或颜色代表其类别。而实例分割在此基础上更进一步。不仅需要确定每个像素的类别，而且还要区分同一类别的不同实例（如图像中的所有车属于同一类别，但每一辆车都是一个实例）。实例分割时每个像素都有两个标签，一个是类别标签，另一个是实例编号。

### 语义分割

样例数据集：[视盘分割数据集](https://bj.bcebos.com/paddlex/datasets/optic_disc_seg.tar.gz)。

语义分割中图像和标签都是图像文件，所以需要通过图像所在文件夹区分图像和标签。所有在`/Dataset Path/JPEGImages/`文件夹下的图像都会被导入，无论图像是否存在标签。所有在`/Dataset Path/Annotations/`文件夹下的图片将被作为标签导入。

示例格式如下：

```shell
Dataset Path
├── Annotations
│   ├── A0001.png
│   ├── B0001.png
│   ├── H0002.png
│   └── ...
├── JPEGImages
│   ├── A0001.jpg
│   ├── B0001.png
│   ├── H0002.bmp
│   └── ...
├── labels.txt
├── test_list.txt
├── train_list.txt
└── val_list.txt

# labels.txt
background -
optic_disk - 128 0 0 // for pesudo color mask, color for each label must be specified
```

#### 数据导入

在语义分割数据集导入过程中，PaddleLabel 将从`labels.txt`中获取标签编号。`labels.txt` 中**第一个标签将被视为背景，并赋标签编号 0**。对于灰度标签，PaddleLabel 会将标签中的像素灰度值与标签编号匹配。而对于伪彩色标签，PaddleLabel 会将每个像素的颜色与`labels.txt`中指定的颜色进行匹配。如果掩膜中有标注没有对应的标签，导入将失败。

标签图像通常用 PNG 格式。 PaddleLabel 在确定图像和标签对应关系时会去掉所有文件拓展名，同名的图像和标签为一组。如果存在多长图像对应一个标签（如图像 image.png 和 image.webp 都对应标签 image.png），导入将会失败。

#### 数据导出

在导出过程中，**`labels.txt`的第一行固定是背景类**。掩膜图像中的值遵循与导入时相同的规则。对于灰度标签，输出将是一个单通道图像，灰度值对应分类标签。对于伪彩色标签，输出将是一个三通道图像，标签颜色作为每个像素的颜色。

对于不同格式的导出，多边形格式就只会保存多边形，而不会保存掩膜；反之只保存掩膜不保存多边形，多边形将转换成掩膜进行保存。因此不建议您在选择了多边形标注或掩膜标注之后仍然将两种标注形式混用。

### 实例分割

实例分割中导入和导出掩膜的过程与语义分割类似，区别是标签由单通道或三通道变为二通道，格式由 png 变为 tiff。tiff 标签中第一个通道（下标 0）是类别标签，第二个通道（下标 1）是实例编号。

用[Napari](https://napari.org/)查看这种标签很方便。可以按照[官方文档](https://napari.org/#installation)进行安装。然后参照下面的步骤使用：

- 打开图像：
  ![image](https://user-images.githubusercontent.com/29757093/184568424-e6925671-8057-4fc2-9a41-dfabcfccdc68.png)
- 打开图像对应的 tiff 掩膜：
  ![image](https://user-images.githubusercontent.com/29757093/184568545-eb9365b1-4c8f-400c-a7d8-876a97d07a6e.png)
- 右键单击掩膜图层，点击`Split Stack`，标签由一个图层变为两个：
  ![image](https://user-images.githubusercontent.com/29757093/184568622-30128ff2-9cd8-4963-940f-fa54f6077f17.png)
  ![image](https://user-images.githubusercontent.com/29757093/184568735-51da72f7-64fc-43f1-9ac5-28f34511d460.png)
- 在两个标签图层中任一个上右键单击，选择`Convert to Label`，将灰度标签转为伪彩色，方便查看：
  ![image](https://user-images.githubusercontent.com/29757093/184569020-2c1ac1fa-3b0f-47da-9b0d-07936b827c53.png)
  ![image](https://user-images.githubusercontent.com/29757093/184569082-ed76582b-abed-4b85-9290-a0e0a1228f25.png)
- 两个图层中，layer 1为类别掩膜，layer 0为实例掩膜
  ![image](https://user-images.githubusercontent.com/29757093/184569144-a1b11662-7615-4251-a7a7-6f665443d67e.png)
  ![image](https://user-images.githubusercontent.com/29757093/184569158-302cc9ba-1669-4f18-8e91-effddd642778.png)

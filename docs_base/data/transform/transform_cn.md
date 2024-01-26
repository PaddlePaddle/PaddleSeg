简体中文|[English](transform.md)

# LabelMe分割数据标注

本文档简要介绍使用LabelMe软件进行分割数据标注，并将标注数据转换为PaddleSeg和PaddleX支持的格式。

## 1. 安装LabelMe

LabelMe支持在Windows/macOS/Linux三个系统上安装。

在Python3环境下，执行如下命令，可以快速安装LabelMe。
```
pip install labelme
```

LabelMe详细的安装和使用流程，可以参照[官方指南](https://github.com/wkentaro/labelme)。

## 2. 使用LabelMe

### 2.1 启动LabelMe

在电脑终端输入`labelme`，稍等会出现LableMe的交互界面。

<div align="center">
<img src="../image/image-1.png"  width = "600" />  
<p>LableMe交互界面</p>
</div>

点击左上角`File`：
* 勾选`Save Automatically`，设置软件自动保存标注json文件，避免需要手动保存
* 取消勾选`Save With Image Data`，设置标注json文件中不保存data数据

<div align="center">
<img src="https://github.com/PaddlePaddle/PaddleSeg/assets/52520497/935090d4-7b4f-4afc-b878-5e2b6c8dd2a8"  width = "600" />  
<p>LableMe设置</p>
</div>


### 2.2 预览已标注图片（可选）

执行如下命令，clone下载LabelMe的代码。
```
git clone https://github.com/wkentaro/labelme.git
```

在LabelMe交互界面上点击`OpenDir`，选择`<path/to/labelme>/examples/semantic_segmentation/data_annotated`目录（`<path/to/labelme>`为clone下载的`labelme`的路径），打开后可以显示的是语义分割的真值标注。

<div align="center">
<img src="../image/image-2.png"  width = "600" />  
<p>已标注图片的示意图</p>
</div>


### 2.3 标注图片

将所有待标注图片保存在一个目录下，点击`OpenDir`打开待标注图片所在目录。

点击`Create Polygons`，沿着前景目标的边缘画闭合的多边形，然后输入或者选择目标的类别。

<div align="center">
<img src="../image/image-3.png"  width = "600" />  
<p>标注单个目标的示意图</p>
</div>

通常情况下，大家只需要标注前景目标并设置标注类别，其他像素默认作为背景。如果大家需要手动标注背景区域，**类别必须设置为`_background_`**，否则格式转换会有问题。

比如针对有空洞的目标，在标注完目标外轮廓后，再沿空洞边缘画多边形，并将空洞指定为特定类别，如果空洞是背景则指定为`_background_`，示例如下。

<div align="center">
<img src="../image/image-10.jpg"  width = "600" />  
<p>带空洞目标的标注示意图</p>
</div>

如果在标注过程中某个点画错了，可以鼠标右键选择撤销该点；点击`Edit Polygons`可以移动多边形的位置，也可以移动某个点的位置；右击点击类别label，可以选择`Edit Label`修改类别名称。

<div align="center">
<img src="../image/image-4-2.png"  width = "600" />  
<p>修改标注的示意图</p>
</div>


图片中所有目标的标注都完成后，直接选择下一张图片进行标注。（由于勾选`Save Automatically`，不再需要手动点击`Save`保存json文件）

检查标注json文件和图片**存放在同一个文件夹**，而且是一一对应关系，如下图所示。

<div align="center">
<img src="https://github.com/PaddlePaddle/PaddleSeg/assets/52520497/03407e35-f5bf-4312-aecd-0929dff1a984"  width = "400" />  
<p>LableMe产出的标注文件的示意图</p>
</div>


## 3. 数据格式转换

使用PaddleSeg提供的数据转换脚本，将LabelMe标注工具产出的数据格式转换为PaddleSeg和PaddleX所需的数据格式。

运行以下代码进行转换，第一个`input_dir`参数是原始图像和json标注文件的保存目录，第二个`output_dir`参数是转换后数据集的保存目录。

```
python tools/data/labelme2seg.py input_dir output_dir
```

格式转换后的数据集目录结构如下：
```
dataset_dir                     # 根目录
|-- images                      # 原始图像的目录
|   |-- xxx.png(png or other)   # 原始图像
|   |...
|-- annotations                 # 标注图像的目录
|   |-- xxx.png                 # 标注图像
|   |...
|-- class_names.txt             # 数据集的类别名称，背景_background_的类别id是0，其他类别id依次递增
```

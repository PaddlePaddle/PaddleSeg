简体中文|[English](transform.md)
# 数据标注教程

无论是语义分割，全景分割，还是实例分割，我们都需要充足的训练数据。如果你想使用没有标注的原始数据集做分割任务，你必须先为原始图像作出标注。如果你使用的是Cityscapes等已有分割标注的数据集，你可以跳过本步骤。
- 本文档将提供2种标注工具的使用教程：[EISeg](#一、EISeg)、[LabelMe](#二、LabelMe)。

# 一、EISeg
# EISeg

EISeg(Efficient Interactive Segmentation)是基于飞桨开发的一个高效智能的交互式分割标注软件。它涵盖了高精度和轻量级等不同方向的高质量交互式分割模型，方便开发者快速实现语义及实例标签的标注，降低标注成本。 另外，将EISeg获取到的标注应用到PaddleSeg提供的其他分割模型进行训练，便可得到定制化场景的高精度模型，打通分割任务从数据标注到模型训练及预测的全流程。

## 演示

![eiseg-demo](../../images/eiseg_demo.gif)

## <span id = "jump">模型准备</span>

在使用EIseg前，请先下载模型参数。EISeg开放了在COCO+LVIS和大规模人像数据上训练的四个标注模型，满足通用场景和人像场景的标注需求。其中模型结构对应EISeg交互工具中的网络选择模块，用户需要根据自己的场景需求选择不同的网络结构和加载参数。

| 模型类型 | 适用场景 | 模型结构 | 下载地址|
| --- | --- | --- | ---|
| 高精度模型  | 适用于通用场景的图像标注。 |HRNet18_OCR64 | [hrnet18_ocr64_cocolvis](https://bj.bcebos.com/paddleseg/dygraph/interactive_segmentation/ritm/hrnet18_ocr64_cocolvis.pdparams) |
| 轻量化模型  | 适用于通用场景的图像标注。 |HRNet18s_OCR48 | [hrnet18s_ocr48_cocolvis](https://bj.bcebos.com/paddleseg/dygraph/interactive_segmentation/ritm/hrnet18s_ocr48_cocolvis.pdparams) |
| 高精度模型  | 适用于人像标注场景。 |HRNet18_OCR64 | [hrnet18_ocr64_human](https://bj.bcebos.com/paddleseg/dygraph/interactive_segmentation/ritm/hrnet18_ocr64_human.pdparams) |
| 轻量化模型  | 适用于人像标注场景。 |HRNet18s_OCR48 | [hrnet18s_ocr48_human](https://bj.bcebos.com/paddleseg/dygraph/interactive_segmentation/ritm/hrnet18s_ocr48_human.pdparams) |


* 1.安装
`
EISeg提供多种安装方式，其中使用[pip](#PIP)，[conda](#Conda)和[运行代码](#运行代码)方式可兼容Windows，Mac OS和Linux。为了避免环境冲突，推荐在conda创建的虚拟环境中安装。

版本要求:

   * PaddlePaddle >= 2.1.0`

PaddlePaddle安装请参考[官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/windows-pip.html)。

### PIP

pip安装方式如下：

```shell
pip install eiseg
```
pip会自动安装依赖。安装完成后命令行输入：
```shell
eiseg
```
即可运行软件。

### Conda
首先安装Anaconda或Miniconda，过程参考[清华镜像教程](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)。

```shell
conda create -n eiseg python=3.8
conda activate eiseg
conda install qtpy
pip install eiseg
eiseg
```

### Windows exe

EISeg使用[QPT](https://github.com/GT-ZhangAcer/QPT)进行打包。可以从[百度云盘](https://pan.baidu.com/s/1KXJ9PYjbnBgQozZJEJE-bA)（提取码：82z9）下载最新EISeg。解压后双击启动程序.exe即可运行程序。程序第一次运行会初始化安装所需要的包，请稍等片刻。

### 运行代码

首先clone本项目到本地。
```shell
git clone https://github.com/PaddlePaddle/PaddleSeg
cd PaddleSeg/contrib/EISeg
pip install -r requirements.txt
python -m eiseg
```
即可开始执行。



## 界面

![eiseg-ui](../../images/eiseg_ui.png)

EISeg的界面主要有4个部分组成，分别是菜单&工具栏、状态栏、图像显示区和工作区。

- 菜单栏、工具栏和状态栏为PyQt默认。工具栏可拖动到任意位置，或放置上下两端。相关工具的介绍当鼠标悬停时显示在状态栏中。
- 工作区采用QDockWidget，也可以进行拖动，放置到任意位置，或固定在左右两端。工作区主要可以进行相关的设置和切换。
- 图像显示区负责图像的显示和交互。采用QGraphicsView，其中加载QGraphicsScene，图像以Qpixmap的形式加入。在图像加载时会自动根据图像大小计算缩放比例，确保图像完整显示。



* 2.使用

打开软件后，在对项目进行标注前，需要进行如下设置：

1. **模型参数加载**

   ​		选择合适的网络，并加载对应的模型参数。在EISeg中，目前网络分为`HRNet18s_OCR48`和`HRNet18_OCR64`，并分别提供了人像和通用两种模型参数，模型参数下载参见[模型准备](#jump)。在正确加载模型参数后，右下角状态栏会给予说明。若网络参数与模型参数不符，将会弹出警告，此时加载失败需重新加载。正确加载的模型参数会记录在`近期模型参数`中，可以方便切换，并且下次打开软件时自动加载退出时的模型参数。

2. **图像加载**

   ​		打开图像/图像文件夹。当看到主界面图像正确加载，`数据列表`正确出现图像路径即可。

3. **标签添加/加载**

   ​		可以通过`添加标签`新建标签，标签分为4列，分别对应像素值、说明、颜色和删除。新建好的标签可以通过`保存标签列表`保存为txt文件，其他合作者可以通过`加载标签列表`将标签导入。通过加载方式导入的标签，重启软件后会自动加载。

4. **自动保存设置**

   ​		在使用中可以将`自动保存`设置上，设定好文件夹（目前只支持英文路径）即可，这样在使用时切换图像会自动将完成标注的图像进行保存。

当设置完成后即可开始进行标注，默认情况下常用的按键/快捷键有：

| 按键/快捷键           | 功能             |
| --------------------- | ---------------- |
| **鼠标左键**          | **增加正样本点** |
| **鼠标右键**          | **增加负样本点** |
| 鼠标中键              | 平移图像         |
| Ctrl+鼠标中键（滚轮） | 缩放图像         |
| S                     | 切换上一张图     |
| F                     | 切换下一张图     |
| Space（空格）         | 完成标注         |
| Ctrl+Z                | 撤销一次点击     |
| Ctrl+Shift+Z          | 清除全部点击     |
| Ctrl+Y                | 重做一次点击     |
| Ctrl+A                | 打开图像         |
| Shift+A               | 打开文件夹       |
| Ctrl+M                | 加载模型参数     |

# 开发者

[Yuying Hao](https://github.com/haoyuying), [Yizhou Chen](https://github.com/geoyee), [Lin Han](https://github.com/linhandev/), [GT](https://github.com/GT-ZhangAcer), [Zhiliang Yu](https://github.com/yzl19940819)



# 二、LabelMe
* 1.LabelMe的安装

用户在采集完用于训练、评估和预测的图片之后，需使用数据标注工具[LabelMe](https://github.com/wkentaro/labelme)完成数据标注。LabelMe支持在Windows/macOS/Linux三个系统上使用，且三个系统下的标注格式是一样。具体的安装流程请参见[官方安装指南](https://github.com/wkentaro/labelme)。

* 2.LabelMe的使用

打开终端输入`labelme`会出现LableMe的交互界面，可以先预览`LabelMe`给出的已标注好的图片，再开始标注自定义数据集。

![](../image/image-1.png)

<div align="left">
    <p>图1 LableMe交互界面的示意图</p>
 </div>


   * 预览已标注图片  

获取`LabelMe`的源码：

```
git clone https://github.com/wkentaro/labelme
```

终端输入`labelme`会出现LableMe的交互界面，点击`OpenDir`打开`<path/to/labelme>/examples/semantic_segmentation/data_annotated`，其中`<path/to/labelme>`为克隆下来的`labelme`的路径，打开后示意的是语义分割的真值标注。

![](../image/image-2.png)

<div align="left">
    <p>图2 已标注图片的示意图</p>
 </div>



   * 开始标注

请按照下述步骤标注数据集：

​		(1)   点击`OpenDir`打开待标注图片所在目录，点击`Create Polygons`，沿着目标的边缘画多边形，完成后输入目标的类别。在标注过程中，如果某个点画错了，可以按撤销快捷键可撤销该点。Mac下的撤销快捷键为`command+Z`。

![](../image/image-3.png)

<div align="left">
    <p>图3 标注单个目标的示意图</p>
 </div>



​		(2)   右击选择`Edit Polygons`可以整体移动多边形的位置，也可以移动某个点的位置；右击选择`Edit Label`可以修改每个目标的类别。请根据自己的需要执行这一步骤，若不需要修改，可跳过。

![](../image/image-4-2.png)

<div align="left">
    <p>图4 修改标注的示意图</p>
 </div>



​		(3)   图片中所有目标的标注都完成后，点击`Save`保存json文件，**请将json文件和图片放在同一个文件夹里**，点击`Next Image`标注下一张图片。

LableMe产出的真值文件可参考我们给出的[文件夹](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.8.0/docs/annotation/labelme_demo)。

![](../image/image-5.png)

<div align="left">
    <p>图5 LableMe产出的真值文件的示意图</p>
 </div>



 **Note：**

 对于中间有空洞的目标的标注方法：在标注完目标轮廓后，再沿空洞区域边缘画多边形，并将其指定为其他类别，如果是背景则指定为`_background_`。如下：

![](../image/image-10.jpg)

 <div align="left">
    <p>图6 带空洞目标的标注示意图</p>
 </div>



* 3.数据格式转换

最后用我们提供的数据转换脚本将上述标注工具产出的数据格式转换为模型训练时所需的数据格式。

* 经过数据格式转换后的数据集目录结构如下：

 ```
 my_dataset                 # 根目录
 |-- annotations            # 数据集真值
 |   |-- xxx.png            # 像素级别的真值信息
 |   |...
 |-- class_names.txt        # 数据集的类别名称
 |-- xxx.jpg(png or other)  # 数据集原图
 |-- ...
 |-- xxx.json               # 标注json文件
 |-- ...

 ```

![](../image/image-6.png)

<div align="left">
    <p>图7 格式转换后的数据集目录的结构示意图</p>
 </div>



* 4.运行以下代码，将标注后的数据转换成满足以上格式的数据集：

```
  python tools/labelme2seg.py <PATH/TO/LABEL_JSON_FILE>
```

其中，`<PATH/TO/LABEL_JSON_FILE>`为图片以及LabelMe产出的json文件所在文件夹的目录，同时也是转换后的标注集所在文件夹的目录。

我们已内置了一个标注的示例，可运行以下代码进行体验：

```
python tools/labelme2seg.py docs/annotation/labelme_demo/
```

转换得到的数据集可参考我们给出的[文件夹](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.8.0/docs/annotation/labelme_demo)。其中，文件`class_names.txt`是数据集中所有标注类别的名称，包含背景类；文件夹`annotations`保存的是各图片的像素级别的真值信息，背景类`_background_`对应为0，其它目标类别从1开始递增，至多为255。

![](../image/image-7.png)

<div align="left">
    <p>图8 格式转换后的数据集各目录的内容示意图</p>
 </div>

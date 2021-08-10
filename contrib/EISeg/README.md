# EISeg

[![Python 3.6](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/) [![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
<!-- [![GitHub release](https://img.shields.io/github/release/Naereen/StrapDown.js.svg)](https://github.com/PaddleCV-SIG/iseg/releases) -->

EISeg(Efficient Interactive Segmentation)是基于飞桨开发的一个高效智能的交互式分割标注软件。在交互式上使用了RITM(Reviving Iterative Training with Mask Guidance for Interactive Segmentation)算法，涵盖了高精度和轻量级等不同方向的高质量交互式分割模型，方便开发者快速实现语义及实例标签的标注，降低标注成本。 另外，将EISeg获取到的标注应用到PaddleSeg提供的其他分割模型进行训练，便可得到定制化场景的高精度模型，打通分割任务从数据标注到模型训练及预测的全流程。

![eiseg_demo](E:\dataFiles\github\PaddleSeg\docs\images\eiseg_demo.gif)

## 模型准备

在使用EIseg前，请先下载模型参数。EISeg开放了在COCO+LVIS和大规模人像数据上训练的四个标注模型，满足通用场景和人像场景的标注需求。其中模型结构对应EISeg交互工具中的网络选择模块，用户需要根据自己的场景需求选择不同的网络结构和加载参数。

| 模型类型 | 适用场景 | 模型结构 | 下载地址|
| --- | --- | --- | ---|
| 高精度模型  | 适用于通用场景的图像标注。 |HRNet18_OCR64 | [hrnet18_ocr64_cocolvis](https://bj.bcebos.com/paddleseg/dygraph/interactive_segmentation/ritm/hrnet18_ocr64_cocolvis.pdparams) |
| 轻量化模型  | 适用于通用场景的图像标注。 |HRNet18s_OCR48 | [hrnet18s_ocr48_cocolvis](https://bj.bcebos.com/paddleseg/dygraph/interactive_segmentation/ritm/hrnet18s_ocr48_cocolvis.pdparams) |
| 高精度模型  | 适用于人像标注场景。 |HRNet18_OCR64 | [hrnet18_ocr64_human](https://bj.bcebos.com/paddleseg/dygraph/interactive_segmentation/ritm/hrnet18_ocr64_human.pdparams) |
| 轻量化模型  | 适用于人像标注场景。 |HRNet18s_OCR48 | [hrnet18s_ocr48_human](https://bj.bcebos.com/paddleseg/dygraph/interactive_segmentation/ritm/hrnet18s_ocr48_human.pdparams) |



## 安装使用

EISeg提供多种安装方式，其中使用[pip](#PIP)，[conda](#Conda)和[运行代码](#运行代码)方式可兼容Windows，Mac OS和Linux。为了避免环境冲突，推荐在conda创建的虚拟环境中安装。

版本要求:

* PaddlePaddle >= 2.1.0

PaddlePaddle安装请参考[官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/windows-pip.html)。

#### PIP

pip安装方式如下：

```shell
pip install eiseg
```
pip会自动安装依赖。安装完成后命令行输入：
```shell
eiseg
```
即可运行软件。

#### Conda
首先安装Anaconda或Miniconda，过程参考[清华镜像教程](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)。

```shell
conda create -n eiseg python=3.8
conda activate eiseg
conda install qtpy
pip install eiseg
eiseg
```

#### Windows exe

EISeg使用[QPT](https://github.com/GT-ZhangAcer/QPT)进行打包。可以从[百度云盘]()（提取码：xxxx）下载最新EISeg。解压后双击启动程序.exe即可运行程序。程序第一次运行会初始化安装所需要的包，请稍等片刻。

## 使用

打开软件后，在对项目进行标注前，需要进行如下设置：

1.  **模型参数加载**

   选择合适的网络，并加载对应的模型参数。在EISeg中，目前网络分为`HRNet18s_OCR48`和`HRNet18_OCR64`，并分别提供了人像和通用两种模型参数。在正确加载模型参数后，右下角状态栏会给予说明。若网络参数与模型参数不符，将会弹出警告，此时加载失败需重新加载。正确加载的模型参数会记录在`近期模型参数`中，可以方便切换，并且下次打开软件时自动加载退出时的模型参数。

2. **图像加载**

   打开图像/图像文件夹。当看到主界面图像正确加载，`数据列表`正确出现图像路径即可。

3. **标签添加/加载**

   添加/加载标签。可以通过`添加标签`新建标签，标签分为4列，分别对应像素值、说明、颜色和删除。新建好的标签可以通过`保存标签列表`保存为txt文件，其他合作者可以通过`加载标签列表`将标签导入。通过加载方式导入的标签，重启软件后会自动加载。

4. **自动保存设置**

   在使用中可以将`自动保存`设置上，设定好文件夹（目前只支持英文路径）即可，这样在使用时切换图像会自动将完成标注的图像进行保存。

当设置完成后即可开始进行标注，默认情况下常用的按键/快捷键有：

| 按键/快捷键           | 功能              |
| --------------------- | ----------------- |
| 鼠标左键              | 增加正样本点      |
| 鼠标右键              | 增加负样本点      |
| 鼠标中键              | 平移图像          |
| Ctrl+鼠标中键（滚轮） | 缩放图像          |
| S                     | 切换上一张图      |
| F                     | 切换下一张图      |
| Space（空格）         | 完成标注/切换状态 |
| Ctrl+Z                | 撤销              |
| Ctrl+Shift+Z          | 清除              |
| Ctrl+Y                | 重做              |
| Ctrl+A                | 打开图像          |
| Shift+A               | 打开文件夹        |
| E                     | 打开快捷键表      |
| Backspace（退格）     | 删除多边形        |
| 鼠标双击（点）        | 删除点            |
| 鼠标双击（边）        | 添加点            |

## 说明

5. 多边形出现后可以通过空格切换状态，以便于在内部进行交互。
6. 标签按住name可以进行拖动，最后生成mask时会根据标签列表从上往下进行覆盖。

## 常见问题

下面列举了一些常见问题及可能的解决方案：

<details> <summary> 选择模型权重后崩溃</summary><pre><code>
提示：EISeg推理过程中使用CPU和GPU版本的Paddle体验差异不是很大，如果GPU版本安装遇到问题可以先使用CPU版本快速尝试。相关安装方法参见[官方安装教程](https://www.paddlepaddle.org.cn/)。
1. Paddle版本低：EISeg中需要Paddle版本为2.1.x，如果版本过低请升级Paddle版本。查看Paddle版本：
<code>python -c "import paddle; print(paddle.__version__)"</code>
升级Paddle：
<code># CPU版本
pip install --upgrade paddlepaddle
# GPU版本
pip install --upgrade paddlepaddle-gpu</code>
2. Paddle安装问题：GPU版本Paddle和Cuda之间版本需要对应，检查安装是否存在问题可以运行。
<code>python -c "import paddle; paddle.utils.run_check()"</code></pre></details>


# 开发者

[Yuying Hao](https://github.com/haoyuying), [Yizhou Chen](https://github.com/geoyee), [Lin Han](https://github.com/linhandev/), [GT](https://github.com/GT-ZhangAcer), [Zhiliang Yu](https://github.com/yzl19940819)

<!-- pip install 'git+https://github.com/openvinotoolkit/datumaro' -->

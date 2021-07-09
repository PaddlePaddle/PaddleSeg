[![Python 3.6](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/) [![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
<!-- [![GitHub release](https://img.shields.io/github/release/Naereen/StrapDown.js.svg)](https://github.com/PaddleCV-SIG/iseg/releases) -->

# EISeg

EISeg(Efficient Interactive Segmentation)是基于飞桨开发的一个高效智能的交互式分割标注软件. 它使用了RITM(Reviving Iterative Training with Mask Guidance for Interactive Segmentation)算法，涵盖了高精度和轻量级等不同方向的高质量交互式分割模型，方便开发者快速实现语义及实例标签的标注，降低标注成本。 另外，将EISeg获取到的标注应用到PaddleSeg提供的其他分割模型进行训练，便可得到定制化场景的高精度模型，打通分割任务从数据标注到模型训练及预测的全流程。



## 模型准备
在使用EIseg前，请先下载模型参数。EISeg开放了在COCO+LVIS和大规模人像数据上训练的四个标注模型，满足通用场景和人像场景的标注需求。其中模型结构对应EISeg交互工具中的网络选择模块，用户需要根据自己的场景需求选择不同的网络结构和加载参数。

| 模型类型 | 适用场景 | 模型结构 | 下载地址|
| --- | --- | --- | ---|
| 高精度模型  | 适用于通用场景的图像标注。 |HRNet18_OCR64 | [hrnet18_ocr64_cocolvis](https://bj.bcebos.com/paddleseg/dygraph/interactive_segmentation/ritm/hrnet18_ocr64_cocolvis.pdparams) |
| 轻量化模型  | 适用于通用场景的图像标注。 |HRNet18_OCR48 | [hrnet18s_ocr48_cocolvis](https://bj.bcebos.com/paddleseg/dygraph/interactive_segmentation/ritm/hrnet18s_ocr48_cocolvis.pdparams) |
| 高精度模型  | 适用于人像标注场景。 |HRNet18_OCR64 | [hrnet18_ocr64_human](https://bj.bcebos.com/paddleseg/dygraph/interactive_segmentation/ritm/hrnet18_ocr64_human.pdparams) |
| 轻量化模型  | 适用于人像标注场景。 |HRNet18_OCR48 | [hrnet18s_ocr48_human](https://bj.bcebos.com/paddleseg/dygraph/interactive_segmentation/ritm/hrnet18s_ocr48_human.pdparams) |




## 安装

EISeg提供多种安装方式，其中使用[pip](#PIP)，[conda](#Conda)和[运行代码](#运行代码)方式可兼容Windows，Mac OS和Linux。为了避免环境冲突，推荐在conda创建的虚拟环境中安装。

版本要求:

* PaddlePaddle >= 2.1.0

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

EISeg使用[QPT](https://github.com/GT-ZhangAcer/QPT)进行打包。可以从[百度云盘](https://pan.baidu.com/s/1K7cbNnlCtfEXcuiamdxjWA)（提取码：82z9）下载目前最新的EISeg，也可从[Releases](https://github.com/PaddleCV-SIG/EISeg/releases)中进行下载。解压后双击启动程序.exe即可运行程序。程序第一次运行会初始化安装所需要的包，请稍等片刻。

### 运行代码

首先clone本项目到本地。
```shell
git clone https://github.com/PaddlePaddle/PaddleSeg
cd PaddleSeg/contrib/eiseg
pip install -r requirements.txt
python -m eiseg
```
即可开始执行。



# 开发者
[YuYing Hao](https://github.com/haoyuying)

[YiZhou Chen](https://github.com/geoyee)

[Lin Han](https://github.com/linhandev/)

[GT](https://github.com/GT-ZhangAcer)

[ZhiLiang Yu](https://github.com/yzl19940819)

[![Python 3.6](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/) [![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
<!-- [![GitHub release](https://img.shields.io/github/release/Naereen/StrapDown.js.svg)](https://github.com/PaddleCV-SIG/iann/releases) -->

# IANN

交互式语义分割标注软件，暂定名iann(Interactive Annotation)。

# 安装

交互式标注过程中需要用到深度学习进行推理，[模型权重文件](./doc/WEIGHT.md)目前需要单独下载。IANN提供多种安装方式，其中使用[pip](#PIP)，[conda](#conda安装)和[运行代码](#运行代码)方式可兼容Windows，Mac OS和Linux。为了避免环境冲突，推荐在conda出的虚拟环境中安装。

## PIP
最简单的安装方式是使用pip
```shell
pip install iann
```
pip会自动安装依赖。安装完成后命令行输入
```shell
iann
```
软件即开始执行。

## Conda
首先安装Anaconda或Miniconda，过程参考[清华镜像教程](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)。
```shell
conda create -n iann python=3.8
conda install qtpy
pip install iann
iann
```

## Windows exe

IANN采用[QPT](https://github.com/GT-ZhangAcer/QPT)进行打包。从[百度云盘](https://pan.baidu.com/s/13Ac_LA2yxPT-tnRqUaYMsQ)（提取码：82z9）下载目前最新的IANN，解压后双击启动程序.exe即可运行程序。程序第一次运行会初始化安装所需要的包，请稍等片刻。

## 运行代码

首先clone本项目到本地。
```shell
git clone https://github.com/paddlecv-sig/iann
cd iann
pip install -r requirements.txt
python -m iann
```
即可开始执行。

注：软件默认安装cpu版Paddle，如需使用GPU版可以按照[Paddle官网教程](https://www.paddlepaddle.org.cn/install/quick)安装。

# 开发者
[JueYing Hao]()

[Lin Han](https://github.com/linhandev/)

[YiZhou Chen](https://github.com/geoyee)

[ZhiLiang Yu](https://github.com/yzl19940819)

<!-- [![Sparkline](https://stars.medv.io/Naereen/badges.svg)](https://stars.medv.io/PaddleCV-SIG/iann) -->


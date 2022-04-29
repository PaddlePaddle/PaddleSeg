# 遥感相关

以下内容为EISeg中遥感垂类相关的文档，主要包括环境配置和功能介绍两大方面。

## 1 环境配置

EISeg中对遥感数据的支持来自GDAL/OGR，GDAL是一个在X/MIT许可协议下的开源栅格空间数据转换库，OGR与其功能类似但主要提供对矢量数据的支持。同时需要安装rasterio。

### 1.1 依赖安装

关于GDAL的安装，可参考如下安装方式：

#### 1.1.1 Windows

Windows用户可以通过[这里](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)下载对应Python和系统版本的二进制文件（*.whl）到本地，以GDAL‑3.3.3‑cp39‑cp39‑win_amd64.whl为例，进入下载目录：

```shell
cd download
```

安装依赖：

```shell
pip install GDAL‑3.3.3‑cp39‑cp39‑win_amd64.whl
```

#### 1.1.2 Linux/Mac安装

Mac用户建议利用conda安装，如下：

```shell script
conda install gdal
```
#### 1.1.3 rasterio 安装

建议用户利用conda安装，如下

```shell script
conda install rasterio
```

## 2 功能介绍

目前EISeg中的遥感垂类功能建设还比较简单，基本完成了GTiff类数据加载、大幅遥感影像切片与合并、地理栅格/矢量数据（GTiff/ESRI Shapefile）导出。并基于各类建筑提取数据集40余万张数据训练了一个建筑分割的交互式模型。

### 2.1 数据加载

目前EISeg仅支持了*.tif/tiff图像后缀的的遥感影像读取，由于训练数据都是来自于RGB三通道的遥感图像切片，因此交互分割也仅在RGB三通道上完成，也就表示EISeg支持多波段数据的波段选择。

当使用EISeg打开GTiff图像时，会获取当前波段数，可通过波段设置的下拉列表进行设置。默认为[b1, b1, b1]。下例展示的是天宫一号多光谱数据设置真彩色：

![yd6fa-hqvvb](https://user-images.githubusercontent.com/71769312/141137443-a327309e-0987-4b2a-88fd-f698e08d3294.gif)

### 2.2 大幅数据切片

目前EISeg对于大幅遥感图像（目前最大尝试为900M，17000*10000大小三通道图像），支持切片预测后合并，其中切片的重叠区域overlap为24。

![140916007-86076366-62ce-49ba-b1d9-18239baafc90](https://user-images.githubusercontent.com/71769312/141139282-854dcb4f-bcab-4ccc-aa3c-577cc52ca385.png)


下面是一副来自谷歌地球的重庆部分地区的切片演示：

![7kevx-q90hv](https://user-images.githubusercontent.com/71769312/141137447-60b305b1-a8ef-4b06-a45e-6db0b1ef2516.gif)

### 2.3 地理数据保存

当打开标注的GTiff图像带有地理参考，可设置EISeg保存时保存为带有地理参考的GTiff和ESRI Shapefile。

- GTiff：已成为GIS和卫星遥感应用的行业图像标准文件。
- ESRI Shapefile：是最常见的的矢量数据格式，Shapefile文件是美国环境系统研究所（ESRI）所研制的GIS文件系统格式文件，是工业标准的矢量数据文件。 所有的商业和开源GIS软件都支持。无处不在的它已成为行业标准。

![82jlu-no59o](https://user-images.githubusercontent.com/71769312/141137726-76457454-5e9c-4ad0-85d6-d03f658ee63c.gif)

### 2.4 遥感标注模型选择

建筑物标注建议使用[static_hrnet18_ocr48_rsbuilding_instance](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18_ocr48_rsbuilding_instance.zip)

简体中文 | [English](README_EN.md)

# EISeg

[![Python 3.6](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/) [![PaddlePaddle 2.2](https://img.shields.io/badge/paddlepaddle-2.2-blue.svg)](https://www.python.org/downloads/release/python-360/) [![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE) [![Downloads](https://pepy.tech/badge/eiseg)](https://pepy.tech/project/eiseg)
<!-- [![GitHub release](https://img.shields.io/github/release/Naereen/StrapDown.js.svg)](https://github.com/PaddleCV-SIG/iseg/releases) -->



## 最新动向

- 新增用于X-Ray胸腔标注模型，该模型基于ResNet50_DeeplabV3+及ResNet18_DeeplabV3+，参考MoCo对比学习的思想，优化其数据增强策略，以匹配X-ray的图像特性，完成多数据源的对比学习预训练，提供可靠的预训练模型参数。
- 新增MRI椎骨标注模型，该模型与广州第一人民医院合作，基于MRSpineSeg训练，可实现一键识别磁共振(MRI)模态和CT模态的矢状位方向的腰椎椎体及其附件。
- 新增铝板瑕疵标注模型，该模型基于百度自建质检数据集训练，可实现常见的铝板瑕疵如黑点，小白线，异物等瑕疵的智能标注。


## 介绍

EISeg(Efficient Interactive Segmentation)是以[RITM](https://github.com/saic-vul/ritm_interactive_segmentation)及[EdgeFlow](https://arxiv.org/abs/2109.09406)算法为基础，基于飞桨开发的一个高效智能的交互式分割标注软件。涵盖了通用、人像、遥感、医疗、工业质检等不同方向的高质量交互式分割模型，方便开发者快速实现语义及实例标签的标注，降低标注成本。 另外，将EISeg获取到的标注应用到PaddleSeg提供的其他分割模型进行训练，便可得到定制化场景的高精度模型，打通分割任务从数据标注到模型训练及预测的全流程。

![4a9ed-a91y1](https://user-images.githubusercontent.com/71769312/141130688-e1529c27-aba8-4bf7-aad8-dda49808c5c7.gif)

## 模型准备

在使用EIseg前，请先下载模型参数。EISeg 0.5.0版本开放了在COCO+LVIS、大规模人像数据、mapping_challenge，Chest X-Ray，MRSpineSeg，LiTS及百度自建质检数据集上训练的7个垂类方向模型，满足通用场景、人像场景、建筑物标注，医疗影像肝脏，胸腔，椎骨及铝板质检的标注需求。其中模型结构对应EISeg交互工具中的网络选择模块，用户需要根据自己的场景需求选择不同的网络结构和加载参数。

| 模型类型   | 适用场景                   | 模型结构       | 模型下载地址                                                     |
| ---------- | -------------------------- | -------------- | ------------------------------------------------------------ |
| 高精度模型 | 通用场景的图像标注 | HRNet18_OCR64  | [static_hrnet18_ocr64_cocolvis](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18_ocr64_cocolvis.zip) |
| 轻量化模型 | 通用场景的图像标注 | HRNet18s_OCR48 | [static_hrnet18s_ocr48_cocolvis](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18s_ocr48_cocolvis.zip) |
| 高精度模型 | 通用图像标注场景      | EdgeFlow | [static_edgeflow_cocolvis](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_edgeflow_cocolvis.zip) |
| 高精度模型 | 人像标注场景      | HRNet18_OCR64  | [static_hrnet18_ocr64_human](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18_ocr64_human.zip) |
| 轻量化模型 | 人像标注场景       | HRNet18s_OCR48 | [static_hrnet18s_ocr48_human](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18s_ocr48_human.zip) |
| 轻量化模型 | 遥感建筑物标注场景    | HRNet18s_OCR48 | [static_hrnet18_ocr48_rsbuilding_instance](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18_ocr48_rsbuilding_instance.zip) |
| 轻量化模型 | 医疗肝脏标注场景       | HRNet18s_OCR48 | [static_hrnet18s_ocr48_lits](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18s_ocr48_lits.zip) |
| 高精度模型\* | x光胸腔标注场景       | Resnet50_Deeplabv3+ | [static_resnet50_deeplab_chest_xray](https://paddleseg.bj.bcebos.com/eiseg/0.5/static_resnet50_deeplab_chest_xray.zip) |
| 高精度模型\* | x光胸腔标注场景       | Resnet18_Deeplabv3+ | [static_resnet18_deeplab_chest_xray](https://paddleseg.bj.bcebos.com/eiseg/0.5/static_resnet18_deeplab_chest_xray.zip) |
| 轻量化模型\* | MRI椎骨图像标注场景       | HRNet18s_OCR48 | [static_hrnet18s_ocr48_MRSpineSeg](https://paddleseg.bj.bcebos.com/eiseg/0.5/static_hrnet18s_ocr48_MRSpineSeg.zip) |
| 轻量化模型\* | 质检铝板瑕疵标注场景      | HRNet18s_OCR48 | [static_hrnet18s_ocr48_aluminium](https://paddleseg.bj.bcebos.com/eiseg/0.5/static_hrnet18s_ocr48_aluminium.zip) |

****NOTE****： 将下载的模型结构`*.pdmodel`及相应的模型参数`*.pdiparams`需要放到同一个目录下，加载模型时只需选择`*.pdiparams`结尾的模型参数位置即可， `*.pdmodel`会自动加载。在使用`EdgeFlow`模型时，请将`使用掩膜`关闭，其他模型使用时请勾选`使用掩膜`。其中，`高精度模型`推荐使用带有显卡的电脑，以便获得更流畅的标注体验。

## 安装使用

EISeg提供多种安装方式，其中使用[pip](#PIP)和[运行代码](#运行代码)方式可兼容Windows，Mac OS和Linux。为了避免环境冲突，推荐在conda创建的虚拟环境中安装。PaddlePaddle安装请参考[官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/windows-pip.html)。

PaddlePaddle及EISeg版本对应关系:

|  EISeg版本  | PaddlePaddle版本  |    备注    |
| :------------------: | :---------------: | :-------: |
|    release/0.5       |       >= 2.2.0    |     使用静态图模式    |
|    release/0.4   |       >= 2.2.0    |    使用静态图模式   |
|    release/0.3       |       >= 2.1.0     |  使用动态图模式 |
|    release/0.2       |        >= 2.1.0      |     使用动态图模式    |

### 克隆到本地

通过git将PaddleSeg克隆到本地：

```shell
git clone https://github.com/PaddlePaddle/PaddleSeg.git
```

安装所需环境（若需要使用到GDAL和SimpleITK请参考**垂类分割**进行安装）：

```shell
pip install -r requirements.txt
```

安装好所需环境后，进入EISeg，可通过直接运行eiseg打开EISeg：

```shell
cd PaddleSeg\EISeg
python -m eiseg
```

或进入eiseg，运行exe.py打开EISeg：

```shell
cd PaddleSeg\EISeg\eiseg
python exe.py
```


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


## 使用

打开软件后，在对项目进行标注前，需要进行如下设置：

1. **模型参数加载**

   根据标注场景，选择合适的网络模型及参数进行加载。目前在EISeg0.4.0中，已经将动态图预测转为静态图预测，全面提升单次点击的预测速度。选择合适的模型及参数下载解压后，模型结构`*.pdmodel`及相应的模型参数`*.pdiparams`需要放到同一个目录下，加载模型时只需选择`*.pdiparams`结尾的模型参数位置即可。静态图模型初始化时间稍长，请耐心等待模型加载完成后进行下一步操作。正确加载的模型参数会记录在`近期模型参数`中，可以方便切换，并且下次打开软件时自动加载退出时的模型参数。

2. **图像加载**

   打开图像/图像文件夹。当看到主界面图像正确加载，`数据列表`正确出现图像路径即可。

3. **标签添加/加载**

   添加/加载标签。可以通过`添加标签`新建标签，标签分为4列，分别对应像素值、说明、颜色和删除。新建好的标签可以通过`保存标签列表`保存为txt文件，其他合作者可以通过`加载标签列表`将标签导入。通过加载方式导入的标签，重启软件后会自动加载。

4. **自动保存设置**

   在使用中可以将`自动保存`设置上，设定好文件夹即可，这样在使用时切换图像会自动将完成标注的图像进行保存。

当设置完成后即可开始进行标注，默认情况下常用的按键/快捷键如下，如需修改可按`E`弹出快捷键修改。

| 部分按键/快捷键       | 功能              |
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

## 特色功能使用说明

- **多边形**

    - 交互完成后使用Space（空格）完成交互标注，此时出现多边形边界；
    - 当需要在多边形内部继续进行交互，则使用空格切换为交互模式，此时多边形无法选中和更改。
    - 多边形可以删除，使用鼠标左边可以对锚点进行拖动，鼠标左键双击锚点可以删除锚点，双击两点之间的边则可在此边添加一个锚点。
    - 打开`保留最大连通块`后，所有的点击只会在图像中保留面积最大的区域，其余小区域将不会显示和保存。

- **保存格式**

    - 打开保存`JSON保存`或`COCO保存`后，多边形会被记录，加载时会自动加载。
    - 若不设置保存路径，默认保存至当前图像文件夹下的label文件夹中。
    - 如果有图像之间名称相同但后缀名不同，可以打开`标签和图像使用相同扩展名`。
    - 还可设置灰度保存、伪彩色保存和抠图保存，见工具栏中7-9号工具。

- **生成mask**

    - 标签按住第二列可以进行拖动，最后生成mask时会根据标签列表从上往下进行覆盖。

- **界面模块**

    - 可在`显示`中选择需要显示的界面模块，正常退出时将会记录界面模块的状态和位置，下次打开自动加载。

- **垂类分割**

    EISeg目前已添加对遥感图像和医学影像分割的支持，使用相关功能需要安装额外依赖。

    - 分割遥感图像请安装GDAL，相关安装及介绍具体详见[遥感标注垂类建设](docs/remote_sensing.md)。
    - 分割医学影像请安装SimpleITK，相关安装及介绍具体详见[医疗标注垂类建设](docs/medical.md)。

- **脚本工具使用**

    EISeg目前提供包括标注转PaddleX数据集、划分COCO格式以及语义标签转实例标签等脚本工具，相关使用方式详见[脚本工具使用](docs/tools.md)。

## 版本更新
- 2022.04.10  **0.5.0**：【1】新增chest_xray模型 【2】新增MRSpineSeg模型 【3】新增铝板质检标注模型
- 2021.12.14  **0.4.1**：【1】修复闪退问题； 【2】新增建筑物遥感标注后处理操作。
- 2021.11.16  **0.4.0**：【1】将动态图预测转换成静态图预测，单次点击速度提升十倍；【2】新增遥感图像标注功能，支持多光谱数据通道的选择；【3】支持大尺幅数据的切片（多宫格）处理；【4】新增医疗图像标注功能，支持读取dicom的数据格式，支持选择窗宽和窗位。
- 2021.09.16  **0.3.0**：【1】初步完成多边形编辑功能，支持对交互标注的结果进行编辑；【2】支持中/英界面；【3】支持保存为灰度/伪彩色标签和COCO格式；【4】界面拖动更加灵活；【5】标签栏可拖动，生成mask的覆盖顺序由上往下覆盖。
- 2021.07.07  **0.2.0**：新增contrib：EISeg，可实现人像和通用图像的快速交互式标注。




## 贡献者

- 感谢[Lin Han](https://github.com/linhandev), [Yizhou Chen](https://github.com/geoyee), [Yiakwy](https://github.com/yiakwy), [GT](https://github.com/GT-ZhangAcer), [Youssef Harby](https://github.com/Youssef-Harby), [Nick Nie](https://github.com/niecongchong) 等开发者及[RITM](https://github.com/saic-vul/ritm_interactive_segmentation) 算法支持。
- 感谢[Weibin Liao](https://github.com/MrBlankness)提供的ResNet50_DeeplabV3+及ResNet18_DeeplabV3+预训练模型。
- 感谢[Junjie Guo](https://github.com/Guojunjie08)及[Jiajun Feng](https://github.com/richarddddd198)在椎骨模型上提供的技术支持。

## 学术引用

如果我们的项目在学术上帮助到你，请考虑以下引用：

```latex
@article{hao2021edgeflow,
  title={EdgeFlow: Achieving Practical Interactive Segmentation with Edge-Guided Flow},
  author={Hao, Yuying and Liu, Yi and Wu, Zewu and Han, Lin and Chen, Yizhou and Chen, Guowei and Chu, Lutao and Tang, Shiyu and Yu, Zhiliang and Chen, Zeyu and others},
  journal={arXiv preprint arXiv:2109.09406},
  year={2021}
}
```

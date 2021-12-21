English | [简体中文](README.md)

# EISeg

[![Python 3.6](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/) 
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Downloads](https://pepy.tech/badge/eiseg)](https://pepy.tech/project/eiseg)
<!-- [![GitHub release](https://img.shields.io/github/release/Naereen/StrapDown.js.svg)](https://github.com/PaddleCV-SIG/iseg/releases) -->


## Latest Developments

- **This document instructs the usage of 0.3.0 EISeg, see [develop](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/EISeg) for the newly released EISeg 0.4.0.**
- Our paper on interactive segmentation named [EdgeFlow](https://arxiv.org/abs/2109.09406) is accepted by ICCV 2021 Workshop.
- We release EISeg 0.3.0 with more functions and support for Polygon editing.

## Introduction

EISeg (Efficient Interactive Segmentation) is an efficient and intelligent interactive segmentation annotation software developed based on PaddlePaddle. It covers a large number of high-quality segmentation models in different directions such as *high-performance* and *lightweight*, providing convenience to the rapid annotation of  semantic and instance labels with reduced cost. In addition, by applying the annotations obtained by EISeg to other segmentation models provided by PaddleSeg for training, high-performance models with customized scenarios can be created, integrating the whole process of segmentation tasks from data annotation to model training and prediction.

![eiseg_demo](../docs/images/eiseg_demo.gif)

## Model Preparation

Please download the model parameters before using EIseg. EISeg provides four annotation models trained on COCO+LVIS and large-scale portrait data to meet the needs of both generic and portrait scenarios. The model architecture corresponds to the network selection module in EISeg interactive tools, and users need to select different network structures and loading parameters in accordance with their own needs.

| Model Type             | Applicable Scenarios                  | Model Architecture | Download Link                                                |
| ---------------------- | ------------------------------------- | ------------------ | ------------------------------------------------------------ |
| High Performance Model | Image annotation in generic scenarios | HRNet18_OCR64      | [hrnet18_ocr64_cocolvis](https://bj.bcebos.com/paddleseg/dygraph/interactive_segmentation/ritm/hrnet18_ocr64_cocolvis.pdparams) |
| Lightweight Model      | Image annotation in generic scenarios | HRNet18s_OCR48     | [hrnet18s_ocr48_cocolvis](https://bj.bcebos.com/paddleseg/dygraph/interactive_segmentation/ritm/hrnet18s_ocr48_cocolvis.pdparams) |
| High Performance Model | Annotation in portrait scenarios      | HRNet18_OCR64      | [hrnet18_ocr64_human](https://bj.bcebos.com/paddleseg/dygraph/interactive_segmentation/ritm/hrnet18_ocr64_human.pdparams) |
| Lightweight Model      | Annotation in portrait scenarios      | HRNet18s_OCR48     | [hrnet18s_ocr48_human](https://bj.bcebos.com/paddleseg/dygraph/interactive_segmentation/ritm/hrnet18s_ocr48_human.pdparams) |



## Installation

EISeg provides multiple ways of installation, among which [pip](#PIP) and [run code](#run code) are compatible with Windows, Mac OS and Linux. It is recommended to install in a virtual environment created by conda for fear of environmental conflicts.

System Requirements:

* PaddlePaddle >= 2.1.0

For more details of the installation of PaddlePaddle, please refer to our [official website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/windows-pip.html)。

### Clone

Clone PaddleSeg to your local system through git:

```shell
git clone https://github.com/PaddlePaddle/PaddleSeg.git
```

Enable EISeg by running eiseg after installing the needed environment:

```shell
cd PaddleSeg\EISeg
python -m eiseg
```

Or you can run exe.py in eiseg:

```shell
cd PaddleSeg\EISeg\eiseg
python exe.py
```

### PIP

Install pip as follows：

```shell
pip install eiseg
```

pip will install dependencies automatically. After that, enter the following at the command line:

```shell
eiseg
```

Now, you can run pip.

## Using

After opening the software, make the following settings before annotating:

1. **Load Model Parameter**

   Select the appropriate network and load the corresponding model parameters. Currently, networks in EISeg are `HRNet18s_OCR48` and `HRNet18_OCR64`, which provide model parameters for portrait and generic scenarios respectively. Successful loading is shown at  the status bar in the lower right corner, while a mismatch between the network parameters and model parameters will trigger a warning of failure load,  requiring to be reloaded. The correctly loaded model parameters will be recorded in `Recent Model Parameters`, which can be easily switched, and the exiting model parameter will be loaded automatically the next time you open the software.

2. **Load Image**

   Open the image or image folder. Things go well when you see that the main screen image is loaded correctly and the image path is rightly shown in `Data List`.

3. **Add/Load Label**

   Add/load labels. New labels can be created by `Add Label`, which are divided into 4 columns corresponding to pixel value, description, color and deletion. The newly created labels can be saved as txt files by `Save Label List`, and other collaborators can import labels by `Load Label List`. Labels imported by loading will be loaded automatically after restarting the software.

4. **Autosave**

   You can choose the right folder and have the `autosave` set up, so that the annotated image will be saved automatically when switching images.

Start the annotation when the above are all set up. Here are the commonly used keys/shortcut keys by default, press `E` to modify them as you need.

| Keys/Shortcut Keys                | Function                       |
| --------------------------------- | ------------------------------ |
| Left Mouse Button                 | Add Positive Sample Points     |
| Right Mouse Button                | Add Negative Sample Points     |
| Middle Mouse Button               | Image Panning                  |
| Ctrl+Middle Mouse Button（wheel） | Image Zooming                  |
| S                                 | Previous Image                 |
| F                                 | Next Image                     |
| Space                             | Finish Annotation/Switch State |
| Ctrl+Z                            | Undo                           |
| Ctrl+Shift+Z                      | Clear                          |
| Ctrl+Y                            | Redo                           |
| Ctrl+A                            | Open Image                     |
| Shift+A                           | Open Folder                    |
| E                                 | Open Shortcut Key List         |
| Backspace                         | Delete Polygon                 |
| Double Click（point）             | Delete Point                   |
| Double Click（edge）              | Add Point                      |

## Instruction of New Functions

- **Polygon**

1. Click Space key to complete interactive annotation, then appears the polygon boundary; when you need to continue the interactive process inside the polygon, click Space to switch to interactive mode so the polygon cannot be selected and changed.
2. The polygon can be dragged and deleted. Use the left mouse to drag the anchor point, double-click the anchor point to delete it, and double-click a side to add an anchor point.
3. With `Keep Maximum Connected Blocks` on, only the largest area will remain in the image, the rest of the small areas will not be displayed and saved.

- **Save Format**

1. Polygons will be recorded and automatically loaded after setting `JSON Save` or `COCO Save`.
2. With no specified save path, the image is save to the label folder under the current image folder by default.
3. If there are images with the same name but different suffixes, you can open `labels and images with the same extensions`.
4. You can also save as grayscale,  pseudo-color or matting image, see tools 7-9 in the toolbar

- **Generate mask**

1. Labels can be dragged by holding down the second column, and the final generated mask will be overwritten from top to bottom according to the label list.

- **Interface Module**

1. You can select the interface module to be presented in `Display`, and the normal exit status and location of the interface module will be recorded, and loaded automatically when you open it next time.

## Version Updates

- 2021.09.16  **0.3.0**：【1】Complete the function of polygon editing with support for editing the results of interactive annotation；【2】Support CH/EN interface；【3】Support saving as grayscale/pseudo-color labels and COCO format；【4】More flexible interface dragging；【5】Achieve the dragging of label bar, and the generated mask is overwritten from top to bottom.
- 2021.07.07  **0.2.0**: Newly added contrib：EISeg，which enables rapid interactive annotation of portrait and generic images.

## Contributors

Our gratitude goes to Developers including [Lin Han](https://github.com/linhandev/), [Yizhou Chen](https://github.com/geoyee), [Yiakwy](https://github.com/yiakwy), [GT](https://github.com/GT-ZhangAcer) and the support of [RITM](https://github.com/saic-vul/ritm_interactive_segmentation).

## Citation

If you find our project useful in your research, please consider citing ：

```latex
@article{hao2021edgeflow,
  title={EdgeFlow: Achieving Practical Interactive Segmentation with Edge-Guided Flow},
  author={Hao, Yuying and Liu, Yi and Wu, Zewu and Han, Lin and Chen, Yizhou and Chen, Guowei and Chu, Lutao and Tang, Shiyu and Yu, Zhiliang and Chen, Zeyu and others},
  journal={arXiv preprint arXiv:2109.09406},
  year={2021}
}
```

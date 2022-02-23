English | [简体中文](README.md)


# EISeg

[![Python 3.6](https://camo.githubusercontent.com/75b8738e1bdfe8a832711925abbc3bd449c1e7e9260c870153ec761cad8dde40/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f707974686f6e2d332e362b2d626c75652e737667)](https://www.python.org/downloads/release/python-360/) [![PaddlePaddle 2.2](https://camo.githubusercontent.com/f792707056617d58db17dca769c9a62832156e183b6eb29dde812b34123c2b18/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f706164646c65706164646c652d322e322d626c75652e737667)](https://www.python.org/downloads/release/python-360/) [![License](https://camo.githubusercontent.com/9330efc6e55b251db7966bffaec1bd48e3aae79348121f596d541991cfec8858/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6c6963656e73652d417061636865253230322d626c75652e737667)](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/EISeg/LICENSE) [![Downloads](https://camo.githubusercontent.com/d3d7e08bac205f34cee998959f85ffcbe6a9aca4129c20d7a5ec449848826d48/68747470733a2f2f706570792e746563682f62616467652f6569736567)](https://pepy.tech/project/eiseg)


## Latest Developments

- Our paper on interactive segmentation named [EdgeFlow](https://arxiv.org/abs/2109.09406) is accepted by ICCV 2021 ILDAV.
- Support static graph inference with fully enhanced interaction speed; launch the latest EISeg 0.4.0 that newly adds remote sensing, medical labeling, and square division labeling.

## Introduction

EISeg (Efficient Interactive Segmentation), built on [RITM](https://github.com/saic-vul/ritm_interactive_segmentation) and [EdgeFlow](https://arxiv.org/abs/2109.09406), is an efficient and intelligent interactive segmentation annotation software developed based on PaddlePaddle. It covers a large number of high-quality segmentation models in different directions such as general scenarios, portrait, remote sensing, medical treatment, etc., providing convenience to the rapid annotation of semantic and instance labels with reduced cost. In addition, by applying the annotations obtained by EISeg to other segmentation models provided by PaddleSeg for training, high-performance models with customized scenarios can be created, integrating the whole process of segmentation tasks from data annotation to model training and inference.

[![4a9ed-a91y1](https://user-images.githubusercontent.com/71769312/141130688-e1529c27-aba8-4bf7-aad8-dda49808c5c7.gif)](https://user-images.githubusercontent.com/71769312/141130688-e1529c27-aba8-4bf7-aad8-dda49808c5c7.gif)

## Model Preparation

Please download the model parameters before using EIseg. EISeg 0.4.0 provides four direction models trained on COCO+LVIS, large-scale portrait data, mapping_challenge, and LiTS(Liver Tumor Segmentation Challenge) to meet the labeling needs of generic and portrait scenarios as well as architecture and liver in medical images. The model architecture corresponds to the network selection module in EISeg interactive tools, and users need to select different network structures and loading parameters in accordance with their own needs.

| Model Type             | Applicable Scenarios                     | Model Architecture | Download Link                                                |
| ---------------------- | ---------------------------------------- | ------------------ | ------------------------------------------------------------ |
| High Performance Model | Image annotation in generic scenarios    | HRNet18_OCR64      | [static_hrnet18_ocr64_cocolvis](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18_ocr64_cocolvis.zip) |
| Lightweight Model      | Image annotation in generic scenarios    | HRNet18s_OCR48     | [static_hrnet18s_ocr48_cocolvis](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18s_ocr48_cocolvis.zip) |
| High Performance Model | Annotation in portrait scenarios         | HRNet18_OCR64      | [static_hrnet18_ocr64_human](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18_ocr64_human.zip) |
| Lightweight Model      | Annotation in portrait scenarios         | HRNet18s_OCR48     | [static_hrnet18s_ocr48_human](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18s_ocr48_human.zip) |
| High Performance Model | Image annotation in generic scenarios    | EdgeFlow           | [static_edgeflow_cocolvis](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_edgeflow_cocolvis.zip) |
| Lightweight Model      | Annotation of remote sensing building    | HRNet18s_OCR48     | [static_hrnet18_ocr48_rsbuilding_instance](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18_ocr48_rsbuilding_instance.zip) |
| Lightweight Model      | Annotation of liver in medical scenarios | HRNet18s_OCR48     | [static_hrnet18s_ocr48_lits](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18s_ocr48_lits.zip) |

**NOTE**： The downloaded model structure `*.pdmodel` and the corresponding model parameters `*.pdiparams` should be put into the same directory. When loading the model, you only need to decide the location of the model parameter at the end of `*.pdiparams`, and `*.pdmodel` will be loaded automatically. When using `EdgeFlow` model, please turn off `Use Mask`, and check `Use Mask` when adopting other models.

## Installation

EISeg provides multiple ways of installation, among which [pip](#PIP) and [run code](#run code) are compatible with Windows, Mac OS and Linux. It is recommended to install in a virtual environment created by conda for fear of environmental conflicts.

Version Requirements:

- PaddlePaddle >= 2.2.0

For more details of the installation of PaddlePaddle, please refer to our [official website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/windows-pip.html).

### Clone

Clone PaddleSeg to your local system through git:

```
git clone https://github.com/PaddlePaddle/PaddleSeg.git
```

Install the required environment (if you need to use GDAL and SimpleITK, please refer to **Vertical Segmentation** for installation).

```
pip install -r requirements.txt
```

Enable EISeg by running eiseg after installing the needed environment:

```
cd PaddleSeg\EISeg
python -m eiseg
```

Or you can run exe.py in eiseg:

```
cd PaddleSeg\EISeg\eiseg
python exe.py
```

### PIP

Install pip as follows：

```
pip install eiseg
```

pip will install dependencies automatically. After that, enter the following at the command line:

```
eiseg
```

Now, you can run pip.

## Using

After opening the software, make the following settings before annotating:

1. **Load Model Parameter**

   Select the appropriate network and load the corresponding model parameters. EISeg0.4.0 witnesses the conversion of dynamic graph inference to static one and comprehensive improvements of the inference speed of a single click. After downloading and decompressing the right model and parameters, the model structure `*.pdmodel` and the corresponding model parameters `*.pdiparams` should be put into the same directory, and only the location of the model parameter at the end of `*.pdiparams`need to be selected when loading the model. The initialization of the static model takes some time, please wait patiently until the model is loaded. The correctly loaded model parameters will be recorded in `Recent Model Parameters`, which can be easily switched, and the exiting model parameter will be loaded automatically the next time you open the software.

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

  - Click Space key to complete interactive annotation, then appears the polygon boundary.
  - When you need to continue the interactive process inside the polygon, click Space to switch to interactive mode so the polygon cannot be selected and changed.
  - The polygon can be deleted. Use the left mouse to drag the anchor point, double-click the anchor point to delete it, and double-click a side to add an anchor point.
  - With `Keep Maximum Connected Blocks` on, only the largest area will remain in the image, the rest of the small areas will not be displayed and saved.

- **Save Format**

  - Polygons will be recorded and automatically loaded after setting `JSON Save` or `COCO Save`.
  - With no specified save path, the image is save to the label folder under the current image folder by default.
  - If there are images with the same name but different suffixes, you can open `labels and images with the same extensions`.
  - You can also save as grayscale,  pseudo-color or matting image, see tools 7-9 in the toolbar.

- **Generate mask**

  - Labels can be dragged by holding down the second column, and the final generated mask will be overwritten from top to bottom according to the label list.

- **Interface Module**

  - You can select the interface module to be presented in `Display`, and the normal exit status and location of the interface module will be recorded, and loaded automatically when you open it next time.

- **Vertical Segmentation**

  EISeg now supports remote sensing images and medical images segmentation, and additional dependencies need to be installed for their functioning.

  - Install GDAL for remote sensing image segmentation, please refer to [Remote Sensing Segmentation](docs/remote_sensing_en.md)。
  - Install SimpleITK for medical images segmentation, please refer to [Medical Image Segmentation](docs/medical_en.md)。

- **Scripting Tool**

  EISeg currently provides scripting tools including annotation to PaddleX dataset, delineation of COCO format and semantic labels to instance labels, etc. See [Scripting Tools Usage](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/EISeg/) for more details.

## Version Updates

- 2021.12.14 **0.4.1**: 【1】Fix the bug of crashing; 【2】Newly add the post-labeling operation of remote sensing building images. 
- 2021.11.16 **0.4.0**: 【1】 Convert dynamic graph inference into static graph inference with ten times' increase in the speed of single click; 【2】 Add the function of remote sensing image labeling, support the selection of multi-spectral data channels; 【3】 Support the processing of slicing (multi squre division) of large size data; 【4】 Add medical image labeling function, support the reading dicom format and the selection of window width and position.
- 2021.09.16  **0.3.0**：【1】Complete the function of polygon editing with support for editing the results of interactive annotation；【2】Support CH/EN interface；【3】Support saving as grayscale/pseudo-color labels and COCO format；【4】More flexible interface dragging；【5】Achieve the dragging of label bar, and the generated mask is overwritten from top to bottom.
- 2021.07.07  **0.2.0**: Newly added contrib：EISeg，which enables rapid interactive annotation of portrait and generic images.

## Acknowledgement

Thanks [Lin Han](https://github.com/linhandev), [Yizhou Chen](https://github.com/geoyee), [Yiakwy](https://github.com/yiakwy), [GT](https://github.com/GT-ZhangAcer), [Youssef Harby](https://github.com/Youssef-Harby), [Nick Nie](https://github.com/niecongchong) for their contributions.

Thanks for the algorithm support of [RITM](https://github.com/saic-vul/ritm_interactive_segmentation).

Thanks for the labelling deisgn of [LabelMe](https://github.com/wkentaro/labelme) and [LabelImg](https://github.com/tzutalin/labelImg).

## Citation

If you find our project useful in your research, please consider citing ：

```
@article{hao2021edgeflow,
  title={EdgeFlow: Achieving Practical Interactive Segmentation with Edge-Guided Flow},
  author={Hao, Yuying and Liu, Yi and Wu, Zewu and Han, Lin and Chen, Yizhou and Chen, Guowei and Chu, Lutao and Tang, Shiyu and Yu, Zhiliang and Chen, Zeyu and others},
  journal={arXiv preprint arXiv:2109.09406},
  year={2021}
}
```

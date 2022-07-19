[Chinese (Simplified)](README.md) | English

<div align="center">

<p align="center">
  <img src="https://user-images.githubusercontent.com/35907364/179460858-7dfb19b1-cabf-4f8a-9e81-eb15b6cc7d5f.png" align="middle" alt="LOGO" width = "500" />
</p>

**An Efficient Interactive Segmentation Tool based on [PaddlePaddle](https://github.com/paddlepaddle/paddle).**

[![Python 3.6](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/) [![PaddlePaddle 2.2](https://img.shields.io/badge/paddlepaddle-2.2-blue.svg)](https://www.python.org/downloads/release/python-360/) [![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE) [![Downloads](https://pepy.tech/badge/eiseg)](https://pepy.tech/project/eiseg)

</div>

<div align="center">
<img src="https://user-images.githubusercontent.com/35907364/179468849-c21927cf-d209-423f-879f-335e4ad644aa.gif"/>  
</div>

## <img src="../docs/images/seg_news_icon.png" width="20"/> Latest Developments
* [2022-07-20] :fire: EISeg 1.0 is released!
  - Added  interactive video object segmentation for general scenes, this work is based on EISeg interactive segmentation model and [MiVOS](https://github.com/hkchengrex/MiVOS).
  - Added 3D segmentation capabilities for abdominal multi-organ and CT vertebral data, and provides 3D visualization tools. For details, please refer to [3D Anotations](docs/video.md).

## <img src="https://user-images.githubusercontent.com/48054808/157795569-9fc77c85-732f-4870-9be0-99a7fe2cff27.png" width="20"/> Introduction

EISeg (Efficient Interactive Segmentation) is an efficient and intelligent interactive segmentation annotation software developed based on PaddlePaddle. It covers a large number of high-quality segmentation models in different directions such as generic scenarios, portrait, remote sensing, medical treatment, video, etc., providing convenience to the rapid annotation of semantic and instance labels with reduced cost. In addition, by applying the annotations obtained by EISeg to other segmentation models provided by PaddleSeg for training, high-performance models with customized scenarios can be created, integrating the whole process of segmentation tasks from data annotation to model training and inference.

[![4a9ed-a91y1](https://user-images.githubusercontent.com/71769312/141130688-e1529c27-aba8-4bf7-aad8-dda49808c5c7.gif)](https://user-images.githubusercontent.com/71769312/141130688-e1529c27-aba8-4bf7-aad8-dda49808c5c7.gif)

## <img src="../docs/images/teach.png" width="20"/> Tutorials
* [Installation](docs/install_en.md)
* [Image Annotation](docs/image_en.md)
* [Video Annotation](docs/video_en.md)
* [Remote Sensing](docs/remote_sensing_en.md)
* [Medical Treatment](docs/medical_en.md)

## <img src="../docs/images/anli.png" width="20"/> Version Updates

- 2022.07.20  **1.0.0**：【1】Add the ability of interactive video object segmentation. 【2】Add 3D annotation model for abdominal multi-organ【3】Added 3D annotation model for  CT vertebra.
- 2022.04.10  **0.5.0**: 【1】Add chest_xray interactive model;【2】Add MRSpineSeg interactive model;【3】Add industrial quality inspection model;【4】Fix geo-transform / CRS error when shapefile saved.
- 2021.12.14 **0.4.1**: 【1】Fix the bug of crashing; 【2】Newly add the post-labeling operation of remote sensing building images.
- 2021.11.16 **0.4.0**: 【1】 Convert dynamic graph inference into static graph inference with ten times' increase in the speed of single click; 【2】 Add the function of remote sensing image labeling, support the selection of multi-spectral data channels; 【3】 Support the processing of slicing (multi squre division) of large size data; 【4】 Add medical image labeling function, support the reading dicom format and the selection of window width and position.
- 2021.09.16  **0.3.0**：【1】Complete the function of polygon editing with support for editing the results of interactive annotation；【2】Support CH/EN interface；【3】Support saving as grayscale/pseudo-color labels and COCO format；【4】More flexible interface dragging；【5】Achieve the dragging of label bar, and the generated mask is overwritten from top to bottom.
- 2021.07.07  **0.2.0**: Newly added contrib：EISeg，which enables rapid interactive annotation of portrait and generic images.

## Contributors

- Our gratitude goes to Developers including [Zhiliang Yu](https://github.com/yzl19940819), [Yizhou Chen](https://github.com/geoyee), [Lin Han](https://github.com/linhandev), [Jinrui Ding](https://github.com/Thudjr), [Yiakwy](https://github.com/yiakwy), [GT](https://github.com/GT-ZhangAcer), [Youssef Harby](https://github.com/Youssef-Harby), [Nick Nie](https://github.com/niecongchong) and the support of [RITM](https://github.com/saic-vul/ritm_interactive_segmentation) and [MiVOS](https://github.com/hkchengrex/MiVOS).
- Thanks for the labelling deisgn of [LabelMe](https://github.com/wkentaro/labelme) and [LabelImg](https://github.com/tzutalin/labelImg).
- Thanks for [Weibin Liao](https://github.com/MrBlankness) to provide the pretrain model of ResNet50_DeeplabV3.
- Thanks for support of [Junjie Guo](https://github.com/Guojunjie08) and [Jiajun Feng](https://github.com/richarddddd198) on MRSpineSeg model.

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

## <img src="../docs/images/chat.png" width="20"/> Community

* If you have any problem or suggestion on EISeg, please send us issues through [GitHub Issues](https://github.com/PaddlePaddle/PaddleSeg/issues).
* Welcome to Join EISeg WeChat Group
<div align="center">
<img src="./docs/images/chat_eiseg.png"  width = "200" />  
</div>

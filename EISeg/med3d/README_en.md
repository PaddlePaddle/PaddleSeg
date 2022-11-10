 English | [简体中文](README.md)

<div align="center">
<p align="center">
  <img src="https://user-images.githubusercontent.com/34859558/188449455-cd4e4099-6e70-44ca-b8de-57bab04c187c.png" align="middle" width = 500" />
</p>

**A easy-to-use, efficient, smart 3D medical image annotation platform**  <img src="https://user-images.githubusercontent.com/34859558/188409382-467c4c45-df5f-4390-ac40-fa24149d4e16.png" width="30"/>

[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)

</div>

##  <img src="https://user-images.githubusercontent.com/34859558/188422593-4bc47c72-866a-4374-b9ed-1308c3453165.png" width="30"/> Brief Introduciton
3D medical data annotation is an important part of training 3D image segmentation models and promotes disease diagnosis and treatment prediction, but 3D medical data annotation relies on time-consuming and laborious manual annotation by professionals. The low labeling efficiency leads to the lack of large-scale labeling data, which seriously hinders the development of medical AI. To solve this problem, we launched EISeg-Med3D, an intelligent annotation platform for 3D medical images based on interactive segmentation.

EISeg-Med3D is a 3D slicer extension for performing **E**fficient **I**nteractive **Seg**mentation on **Med**ical image in **3D** medical images. Users will guide a deep learning model to perform segmentation by providing positive and negative points. It is simple to install, easy to use and accurate, which can achieve ten times efficiency lift compares to manual labelling. At present, our medical annotation provides the try-on experience on the specified [MRI vertebral data](https://aistudio.baidu.com/aistudio/datasetdetail/81211). If there is a need for 3D annotation on other data, you can make a [contact](https://github.com/PaddlePaddle/PaddleSeg/issues/new/choose).


<div align="center">
<p align="center">
  <img src="https://user-images.githubusercontent.com/34859558/188415269-10526530-0415-4632-8223-0e5d755db29c.gif"  align="middle" width = 900"/>
</p>
</div>


## <img src="https://user-images.githubusercontent.com/34859558/188419267-bd117697-7456-4c72-8cbe-1272264d4fe4.png" width="30"/> Feature
* **Efficient**：Each category only needs a few clicks to generate 3d segmentation results, ten times efficient compares to time-consuming and laborious manual annotation.

* **Accurate**：The mIOU can reach 0.85 with only 3 clicks. with the segmentation editor equipped with machine learning algorithm and manual annotation, 100% accuracy is right on your hand.

* **Convenient**：Install our plugin within three steps; labeling results and progress are automatically saved; the transparency of labeling results can be adjusted to improve labeling accuracy; user-friendly interface interaction makes labeling worry-free and hassle-free。

*************

## Contents
0. [News](##News)
1. [EISeg-Med3D Model Introduction](##EISeg-Med3DModelIntroduction)
2. [User Guide](##UserGuide)
3. [TODO](##TODO)
4. [License](##License)
5. [Thanks](##Thanks)


## <img src="https://user-images.githubusercontent.com/34859558/190043516-eed25535-10e8-4853-8601-6bcf7ff58197.png" width="30"/> 最新消息
- [2022-09] EISeg-Med3D is officially released, **a user-friendly, efficient and intelligent 3D medical image annotation platform** including high-precision models on specified vertebral data.

## <img src="https://user-images.githubusercontent.com/34859558/190049708-7a1cee3c-322b-4263-9ed0-23051825b1a6.png" width="30"/> EISeg-Med3D Model
The EISeg-Med 3D model structure is shown in the figure below. We innovatively introduce the 3D model into the medical interactive segmentation, and modify the point sampler module and the click feature  extrator of RITM to be compatible with 3D data, so as to directly label 3D medical images. Compared with 2D interactive annotation on 3D images, our method is more acurate and more efficient.

The overall model includes two parts: click generation module, click feature generation module, click feature and input image fusion and segmentation model:
* Click generation module 3D click sampler: generate the positive and negative click through sampling on the 3D labelled data directly, where the positive point is a random point at the center of the 3D target segment, and the negative point is a random point at the edge of the 3D target segment.
* Click feature generation module 3D feature extractor: After the click is generated, a sphere with radius R is generated on the basis of the original click in the form of disk to expand the feature coverage.
* Fusion of click feature and input image: The input image and the generated click feature are remapped and fused through the convolution block, so that the network obtains information from both the image and the clicks, and performs 3D segmentation on a the assigned area to obtain the annotation result.
* Segmentation model: The segmentation model is the 3D segmentation model Vnet, generates the prediction results shown in the figure, and approximates the real results under the constraints of Dice loss and CE loss. Thus, in the prediction stage, input images and specified clicks generate the desired annotation results.

<p align="center">
<img src="https://user-images.githubusercontent.com/34859558/190861789-793bd9f3-17a8-49d6-a2a7-bce82696d28e.png" width="80.6%" height="20%">
<p align="center">
 EISeg-Med3D Model
</p>
</p>


##  <img src="https://user-images.githubusercontent.com/34859558/188439970-18e51958-61bf-4b43-a73c-a9de3eb4fc79.png" width="30"/> User Guide
The overall process of using EISeg-Med3D is shown in the figure below. We will introduce in the following three steps including environment installation, model and data download and user guide. The steps to use our platform can also be found in the video in the introduction.


<p align="center">
<img src="https://user-images.githubusercontent.com/48357642/187884472-32e6dd36-be7b-4b32-b5d1-c0ccd743e1ca.png" width="60.6%" height="20%">
<p align="center">
 The overall process
</p>
</p>


<p align="center">
<img src="https://user-images.githubusercontent.com/48357642/187884776-470195d6-46d1-4e2b-8403-0cb320f185ec.png" width="80.6%" height="60%">
<p align="center">
The process of AI-assisted labelling
</p>
</p>


### 环境安装

1. Download and install 3D slicer：[Slicer website](https://www.slicer.org/)

2. Download code of EISeg-Med3D：
```
git clone https://github.com/PaddlePaddle/PaddleSeg.git
```

3. Install PaddlePaddle in python interpreter of slicer，refer to [install doc](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html). If you install on Windows with CUDA 11.1 GPU, follow the command here：
```
import sys
import os
sys.executable # 'D:/slicer/Slicer 5.0.3/bin/PythonSlicer.exe'

os.system(f"'{sys.executable}' -m pip install paddlepaddle-gpu==2.3.1.post111 -f  https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html")

```

The final line should output 0. Anything else indicates the installation has failed. If you started 3D Slicer from a terminal, you should be able to see pip install progress there.

<details>
<summary><b> Common FAQ </b></summary>
1. Error after install PaddlePaddle：
<p align="center">
<img src="https://user-images.githubusercontent.com/34859558/189288387-4773c35a-ac8e-421d-bfed-2264ac57cda5.png" width="70.6%" height="20%">
</p>

Enter the corresponding subprocess.py change shell=True in Popen class.

</details>


### Model and Data Downloading
Currently we provide trial experience on the following models and data:
<p align="center">

| Data | Model | Links |
|:-:|:-:|:-:|
| MRI-spine | Interactive Vnet |[pdiparams](https://pan.baidu.com/s/1Dk-PqogeJOiaEGBse3kFOA)-pw: 6ok7 \| [pdmodel](https://pan.baidu.com/s/1daFrC1C2cwCmovvLj5n3QA)-pw: sg80 \| [Spine Data](https://aistudio.baidu.com/aistudio/datasetdetail/81211)|

</p>

### User Guide
#### 1. Load the Extension
* Locate Extension wizard:
<p align="center">
<img src="https://user-images.githubusercontent.com/34859558/188458289-b59dc5e3-34eb-4d40-b18b-ce0b35c066c6.png" width="70.6%" height="20%">
</p>

* Click on "Select Extension"，and choose PaddleSeg/EISeg/med3d directory, and click to load corresponding module.
<p align="center">
<img src="https://user-images.githubusercontent.com/34859558/188458463-066ff0b6-ff80-4d0d-aca0-3b3b12f710ef.png" width="70.6%" height="20%">
</p>

* After loading, switch to EISegMed3D。
<p align="center">
<img src="https://user-images.githubusercontent.com/34859558/188458684-46465fed-fdde-43dd-a97c-5da7678f3f99.png" width="70.6%" height="20%">
</p>


#### 2. Load Model
* Load the downloaded model in ```Model Settings```:
click on the  ```...``` button of ```Model Path```, choose local file of ```.pdodel``` suffix and load ```.pdiparams``` file in ```Param Path``` in the same way.
* Click on ```Load Static Model``` button. And ```Sucessfully loaded model to gpu!``` window will be prompt is the model is loaded successfully.

<p align="center">
<img src="https://user-images.githubusercontent.com/48357642/187881886-e4d99fb4-c697-48a5-8cd7-a5ab83c7791d.PNG" width="70.6%" height="20%">
</p>

#### 3. Load Medical Data
* Click on the button behind ```Data Folder```, choose the folder that you saved your downloaded data. And all of the data under that folder will be loaded and you can see the labelling status of loaded data in ```Progress```.

<p align="center">
<img src="https://user-images.githubusercontent.com/48357642/187882370-6f9a8f21-8a96-4a4c-8451-18d6e608f7e4.PNG" width="70.6%" height="20%">
</p>

#### 4. Switch Between Images.
* Click on the ```Prev Scan``` button to see the previous image.
* Click on the ```Next Scan``` button to see the next image.

<p align="center">
<img src="https://user-images.githubusercontent.com/48357642/187882440-e1c3cc03-b79e-4ad8-9987-20af42c9ae01.PNG" width="70.6%" height="20%">
</p>

#### 5. Start to label
* Click ```Add/Remove``` in ```Segment Editor``` to add or remove the label. You can change the name of added label by double click the label item.
* Choose the label you want to label and click on the ```Positive Point``` or ```Negative Point``` to enter interactive label mode。
* Click on ```Finish Segment``` button to finish annotation of current segment, you can further edit the annotatioin using tools in segment editor or you can repeat previous step to label next category. If you finished the annotation on this case, you can click on the ```Finish Scan``` button.

<p align="center">
<img src="https://user-images.githubusercontent.com/48357642/187882400-8ee24469-6cb7-4c6a-acf8-df0e14e3f2a7.PNG" width="70.6%" height="20%">
</p>

* See [Slicer Segment editor](https://slicer.readthedocs.io/en/latest/user_guide/modules/segmenteditor.html) for using the tool in segment editor.



#### 6. Check Label Progress
* In ```Annotation Progress``` of ```Progress```, you can checkout the labelling progress of loaded images.
* Doule click on one of the image in the chart below ```Annotation Progress``` will jump to the corresponding image.

<p align="center">
<img src="https://user-images.githubusercontent.com/48357642/187882460-0eb0fc86-d9d7-4733-b812-85c62b1b9281.PNG" width="70.6%" height="20%">
</p>

<!-- </details> -->

## <img src="https://user-images.githubusercontent.com/34859558/190046674-53e22678-7345-4bf1-ac0c-0cc99718b3dd.png" width="30"/> TODO
In the future, we want to continue to develop EISeg-Med3D in these aspects, welcome to join our developer team.
- [ ] Work on larger vertebrae datasets and improve generality of our spine model.
- [ ] Develop models trained on multiple organs to obtain models that generalize to multiple organs.

## <img src="https://user-images.githubusercontent.com/34859558/188446853-6e32659e-8939-4e65-9282-68909a38edd7.png" width="30"/> License

EISeg-Med3D is released under the [Apache 2.0 license](LICENSE).


## <img src="https://user-images.githubusercontent.com/34859558/188446803-06c54d50-f2aa-4a53-8e08-db2253df52fd.png" width="30"/> Attribution

Thanks to  <a href="https://www.flaticon.com/free-icons/idea" title="idea icons"> Idea icons created by Vectors Market - Flaticon</a> for facsinating icons.

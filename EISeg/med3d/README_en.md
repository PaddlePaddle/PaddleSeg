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
3D medical data annotation is widely used in medical image segmentation and promotes disease diagnosis and treatment prediction. However, medical data annotation relies on professionals, and traditional manual annotation is time-consuming and laborious. The lack of annotation data greatly hinders the development downstream applications. Therefore, it is urgent to solve the problem of labeling efficiency. To this end, we present EISeg-Med3D, an intelligent annotation platform for 3D medical images based on interactive segmentation.


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

##  <img src="https://user-images.githubusercontent.com/34859558/188439970-18e51958-61bf-4b43-a73c-a9de3eb4fc79.png" width="30"/> 使用指南
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

* Click on "Select Extension"，and choose contrib/SlicerEISegMed3D directory, and click to load corresponding module.
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

## <img src="https://user-images.githubusercontent.com/34859558/188446853-6e32659e-8939-4e65-9282-68909a38edd7.png" width="30"/> License

EISeg-Med3D is released under the [Apache 2.0 license](LICENSE).


## <img src="https://user-images.githubusercontent.com/34859558/188446803-06c54d50-f2aa-4a53-8e08-db2253df52fd.png" width="30"/> Attribution

Thanks to  <a href="https://www.flaticon.com/free-icons/idea" title="idea icons"> Idea icons created by Vectors Market - Flaticon</a> for facsinating icons.

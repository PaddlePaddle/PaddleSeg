简体中文 | [English](README_en.md)

<div align="center">
<p align="center">
  <img src="https://user-images.githubusercontent.com/34859558/188449455-cd4e4099-6e70-44ca-b8de-57bab04c187c.png" align="middle" width = 500" />
</p>

**专注用户友好、高效、智能的3D医疗图像标注平台**  <img src="https://user-images.githubusercontent.com/34859558/188409382-467c4c45-df5f-4390-ac40-fa24149d4e16.png" width="30"/>

[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)

</div>

##  <img src="https://user-images.githubusercontent.com/34859558/188422593-4bc47c72-866a-4374-b9ed-1308c3453165.png" width="30"/> 简介
3D 医疗数据标注广泛用于医疗图像分割并促进疾病诊断和治疗预测，但医疗数据标注依赖专业人士，而传统的手工标注又存在费时费力的问题，标注数据的缺乏极度阻碍了 3D 医疗数据标注的下游应用发展。因此解决标注效率亟待解决。为此我们推出了基于交互式分割的3D医疗图像智能标注平台 EISeg-Med3D。

EISeg-Med3D 是一个用于智能医学图像分割的 3D Slicer 插件，通过使用训练的交互式分割 AI 模型来进行交互式医学图像标注。它安装简单、使用方便，结合高精度的预测模型，可以获取比手工标注数十倍的效率提升。目前我们的医疗标注提供了在指定的 [MRI 椎骨数据](https://aistudio.baidu.com/aistudio/datasetdetail/81211)上的使用体验，如果有其他数据上的3D标注需求，可以[联系我们](https://github.com/PaddlePaddle/PaddleSeg/issues/new/choose)。


<div align="center">
<p align="center">
  <img src="https://user-images.githubusercontent.com/34859558/188415269-10526530-0415-4632-8223-0e5d755db29c.gif"  align="middle" width = 900"/>
</p>

</div>


## <img src="https://user-images.githubusercontent.com/34859558/188419267-bd117697-7456-4c72-8cbe-1272264d4fe4.png" width="30"/> 特性
* **高效**：每个类别只需数次点击直接生成3d分割结果，从此告别费时费力的手工标注。

* **准确**：点击 3 点 mIOU 即可达到0.85，配合搭载机器学习算法和手动标注的标注编辑器，精度 100% 不是梦。

* **便捷**：三步轻松安装；标注结果、进度自动保存；标注结果透明度调整提升标注准确度；用户友好的界面交互，让你标注省心不麻烦。

*************

##  <img src="https://user-images.githubusercontent.com/34859558/188439970-18e51958-61bf-4b43-a73c-a9de3eb4fc79.png" width="30"/> 使用指南
EISeg-Med3D 的使用整体流程如下图所示，我们将按照环境安装、模型下载和使用步骤三部分说明，其中使用步骤也可以参见简介中的视频。


<p align="center">
<img src="https://user-images.githubusercontent.com/48357642/187884472-32e6dd36-be7b-4b32-b5d1-c0ccd743e1ca.png" width="60.6%" height="20%">
<p align="center">
 整体使用流程
</p>
</p>


<p align="center">
<img src="https://user-images.githubusercontent.com/48357642/187884776-470195d6-46d1-4e2b-8403-0cb320f185ec.png" width="80.6%" height="60%">
<p align="center">
智能标注模块流程
</p>
</p>


### 1. 环境安装

1. 下载并安装3D slicer软件：[3D slicer 官网](https://www.slicer.org/)

2. 下载 EISeg-Med3D 代码：
```
git clone https://github.com/PaddlePaddle/PaddleSeg.git
```

3. 安装Paddle，在slicer的python interpreter中，参考[快速安装文档](https://www.paddlepaddle.org.cn/install/quick)安装PaddlePaddle。 比如Windows、CUDA 11.1，安装GPU版本，执行如下命令：
```
import sys
import os
sys.executable # 'D:/slicer/Slicer 5.0.3/bin/PythonSlicer.exe'

os.system(f"'{sys.executable}' -m pip install paddlepaddle-gpu==2.3.1.post111 -f  https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html")

```
<details>
<summary><b> 常见问题 </b></summary>
1. 安装PaddlePaddle之后出现下面报错：
<p align="center">
<img src="https://user-images.githubusercontent.com/34859558/189288387-4773c35a-ac8e-421d-bfed-2264ac57cda5.png" width="70.6%" height="20%">
</p>

进入到报错位置所在的 subprocess.py， 修改Popen类的属性 shell=True 即可。


</details>

<!-- <details>
<summary><b>使用步骤</b></summary> -->
### 2. 模型、数据下载
目前我们提供在下列模型和数据上的试用体验：
<p align="center">

| 数据 | 模型 | 下载链接 |
|:-:|:-:|:-:|
| MRI椎骨数据 | 交互式 Vnet |[pdiparams](https://pan.baidu.com/s/1Dk-PqogeJOiaEGBse3kFOA)-pw: 6ok7 \| [pdmodel](https://pan.baidu.com/s/1daFrC1C2cwCmovvLj5n3QA)-pw: sg80 \| [椎骨数据](https://aistudio.baidu.com/aistudio/datasetdetail/81211)|

</p>

### 3. 使用步骤
#### 1. 加载插件
* 找到 Extension wizard 插件：
<p align="center">
<img src="https://user-images.githubusercontent.com/34859558/188458289-b59dc5e3-34eb-4d40-b18b-ce0b35c066c6.png" width="70.6%" height="20%">
</p>

* 点击 Select Extension，并选择到contrib/MedicalSeg/Med3DAnnotation目录，并点击加载对应模块，等待 Slicer 进行加载。
<p align="center">
<img src="https://user-images.githubusercontent.com/34859558/188458463-066ff0b6-ff80-4d0d-aca0-3b3b12f710ef.png" width="70.6%" height="20%">
</p>

* 加载完后，切换到 EISegMed3D模块。
<p align="center">
<img src="https://user-images.githubusercontent.com/34859558/188458684-46465fed-fdde-43dd-a97c-5da7678f3f99.png" width="70.6%" height="20%">
</p>


#### 2. 加载模型
* 在```Model Settings```中加载保存在本地的模型，点击```Model Path```路径选择框后面的```...```的按钮，选择后缀名为```.pdodel```的本地文件，点击```Param Path```路径选择框后面的```...```的按钮，选择后缀名为```.pdiparams```的本地文件。
* 点击```Load Static Model```按钮，此时会有弹窗提示```Sucessfully loaded model to gpu!```，表示模型已经加载进来。
<p align="center">
<img src="https://user-images.githubusercontent.com/48357642/187881886-e4d99fb4-c697-48a5-8cd7-a5ab83c7791d.PNG" width="70.6%" height="20%">
</p>

#### 3. 加载图像
* 点击```Data Folder```后面的按钮，选择待标注的医学图像文件所在路径后，会自动把该路径下的所有图像全部加载，此时可以在```Progress```中查看加载进来的所有图像以及当前已标注状态。

<p align="center">
<img src="https://user-images.githubusercontent.com/48357642/187882370-6f9a8f21-8a96-4a4c-8451-18d6e608f7e4.PNG" width="70.6%" height="20%">
</p>

#### 4. 开始标注
* 在```Segment Editor```中点击```Add/Remove```按钮便可自行添加标签或是删除标签，添加标签时会有默认命名，也可以双击标签自行给标签命名。
* 添加标签完毕后即可选中某个标签，点击```Positive Point```或是```Negative Point```后的按钮即可开始交互式标注。
* 点击```Finish Segment```按钮，即可结束当前所选标签下的标注，此时可点击左侧的橡皮擦等工具对标注结果进行精修。或者可重复以上步骤进行下一个对象的标注，否则可点击```Finish Scan```按钮，便会切换下一张图像。
<p align="center">
<img src="https://user-images.githubusercontent.com/48357642/187882400-8ee24469-6cb7-4c6a-acf8-df0e14e3f2a7.PNG" width="70.6%" height="20%">
</p>

* 关于精细修改标注工具的使用，详细可见[Slicer Segment editor](https://slicer.readthedocs.io/en/latest/user_guide/modules/segmenteditor.html)

#### 5. 切换图像
* 点击```Prev Scan```按钮可以切换上一张图像到当前视图框内。
* 点击```Next Scan```按钮可以切换下一张图像到当前视图框内。
<p align="center">
<img src="https://user-images.githubusercontent.com/48357642/187882440-e1c3cc03-b79e-4ad8-9987-20af42c9ae01.PNG" width="70.6%" height="20%">
</p>

#### 6. 查看标注进程
* 在```Progress```中的```Annotation Progress```后面的进度条中可以查看当前加载进来的图像标注进程。
* 双击```Annotation Progress```下方表格中某一张图像文件名，便可以自动跳转到所选图像。
<p align="center">
<img src="https://user-images.githubusercontent.com/48357642/187882460-0eb0fc86-d9d7-4733-b812-85c62b1b9281.PNG" width="70.6%" height="20%">
</p>
<!-- </details> -->

## <img src="https://user-images.githubusercontent.com/34859558/188446853-6e32659e-8939-4e65-9282-68909a38edd7.png" width="30"/> License

EISeg-Med3D 的 License 为 [Apache 2.0 license](LICENSE).


## <img src="https://user-images.githubusercontent.com/34859558/188446803-06c54d50-f2aa-4a53-8e08-db2253df52fd.png" width="30"/> 致谢

感谢 <a href="https://www.flaticon.com/free-icons/idea" title="idea icons"> Idea icons created by Vectors Market - Flaticon</a> 给我们提供了好看的图标

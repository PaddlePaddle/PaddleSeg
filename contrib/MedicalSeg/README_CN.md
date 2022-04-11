[English](README.md) | 简体中文

# MedicalSeg
MedicalSeg 是一个简单易使用的 3D 医学图像分割工具包，支持从数据准备到部署的全流程 GPU 加速。目前支持上十种各种数据集，包括 [COVID-19 CT scans](https://www.kaggle.com/andrewmvd/covid19-ct-scans) 和 [MRISpineSeg](https://aistudio.baidu.com/aistudio/datasetdetail/81211) 数据集。下图是我们的框架基于 Vnet 训练之后的可视化结果，其中使用了 [itkwidgets](https://github.com/InsightSoftwareConsortium/itkwidgets)，你也可以使用我们的[visualize.ipynb](visualize.ipynb) 来可视化你的 3D 数据。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://github.com/shiyutang/files/raw/main/ezgif.com-gif-maker%20(1).gif" width="40.5%" height="50%">  
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://github.com/shiyutang/files/raw/main/ezgif.com-gif-maker.gif" width="53.8%" height="90%">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">Segmentation result of our VNet model on COVID-19 CT scans (mDice on evalset is 97.04%) &   MRISpineSeg (16 class mDice on evalset is 89.14%)</div>
</center>


**MedicalSeg is currently under development! If you find any problem using it or want to share any future develop suggestions, please open a github issue or join us by scanning the following wechat QR code.**

<p align="center">
<img src="https://user-images.githubusercontent.com/48433081/162115375-2dba8796-5184-4793-8efa-b142734fe734.png" width="28%" height="20%">
</p>

## Contents
1. [Performance](##Performance)
2. [Demo](##Demo)
3. [Structure](#Structure)
4. [TODO](#TODO)
5. [Acknowledgement](#Acknowledgement)

## Performance

###  1. Accuracy

We successfully validate our framework with [Vnet](https://arxiv.org/abs/1606.04797) on the [COVID-19 CT scans](https://www.kaggle.com/andrewmvd/covid19-ct-scans) and [MRISpineSeg](https://www.spinesegmentation-challenge.com/) dataset. With the lung mask as label, we reached dice coefficient of 97.04% on COVID-19 CT scans. You can download the log to see the result or load the model and validate it by yourself :).

#### **Result on Lung coronavirus** 

| Backbone | Resolution | lr | Training Iters | Dice | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|
|-|128x128x128|0.001|15000|97.04%|[model](https://bj.bcebos.com/paddleseg/paddleseg3d/lung_coronavirus/vnet_lung_coronavirus_128_128_128_15k_1e-3/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/paddleseg3d/lung_coronavirus/vnet_lung_coronavirus_128_128_128_15k_1e-3/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=9db5c1e11ebc82f9a470f01a9114bd3c)|
|-|128x128x128|0.0003|15000|92.70%|[model](https://bj.bcebos.com/paddleseg/paddleseg3d/lung_coronavirus/vnet_lung_coronavirus_128_128_128_15k_3e-4/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/paddleseg3d/lung_coronavirus/vnet_lung_coronavirus_128_128_128_15k_3e-4/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=0fb90ee5a6ea8821c0d61a6857ba4614)|

#### **Result on MRISpineSeg**

| Backbone | Resolution | lr | Training Iters | Dice(20 classes) | Dice(16 classes) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|-|512x512x12|0.1|15000|74.41%| 88.17% |[model](https://bj.bcebos.com/paddleseg/paddleseg3d/mri_spine_seg/vnet_mri_spine_seg_512_512_12_15k_1e-1/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/paddleseg3d/mri_spine_seg/vnet_mri_spine_seg_512_512_12_15k_1e-1/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=36504064c740e28506f991815bd21cc7)|
|-|512x512x12|0.5|15000|74.69%| 89.14% |[model](https://bj.bcebos.com/paddleseg/paddleseg3d/mri_spine_seg/vnet_mri_spine_seg_512_512_12_15k_5e-1/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/paddleseg3d/mri_spine_seg/vnet_mri_spine_seg_512_512_12_15k_5e-1/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/index?id=08b0f9f62ebb255cdfc93fd6bd8f2c06)|


### 2. Speed
We add gpu acceleration in data preprocess using [CuPy](https://docs.cupy.dev/en/stable/index.html). Compared with preprocess data on cpu, acceleration enable us to use about 40% less time in data prepeocessing. The following shows the time we spend in process COVID-19 CT scans.

<center>

| Device | Time(s) |
|:-:|:-:|
|CPU|50.7|
|GPU|31.4( &#8595; 38%)|

</center>


## Demo
This part introduce a easy to use demo on COVID-19 CT scans dataset. This demo is available on our [Aistudio project](https://aistudio.baidu.com/aistudio/projectdetail/3519594) as well. Detailed steps on training and add your own dataset can refer to this [tutorial](documentation/tutorial.md).
- Download our repository.
    ```
    git clone https://github.com/PaddlePaddle/PaddleSeg.git

    cd contrib/MedicalSeg/
    ```
- Install requirements:
    ```
    pip install -r requirements.txt
    ```
- (Optional) Install CuPY if you want to accelerate the preprocess process. [CuPY installation guide](https://docs.cupy.dev/en/latest/install.html)

- Get and preprocess the data:
    - change the GPU setting [here](tools/preprocess_globals.yml) to True if you installed CuPY and want to use GPU to accelerate.
    ```
    python tools/prepare_lung_coronavirus.py
    ```

- Run the train and validation example. (Refer to the following usage to get the correct result.)
   ```
   sh run-vnet.sh
   ```

## Structure
This part shows you the whole picture of our repository, which is easy to expand with different model and datasets. Our file tree is as follows:

```bash
├── configs         # All configuration stays here. If you use our model, you only need to change this and run-vnet.sh.
├── data            # Data stays here.
├── deploy          # deploy related doc and script.
├── medicalseg  
│   ├── core        # the core training, val and test file.
│   ├── datasets  
│   ├── models  
│   ├── transforms  # the online data transforms
│   └── utils       # all kinds of utility files
├── export.py
├── run-unet.sh     # the script to reproduce our project, including training, validate, infer and deploy
├── tools           # Data preprocess including fetch data, process it and split into training and validation set
├── train.py
├── val.py
└── visualize.ipynb # You can try to visualize the result use this file.
```

## TODO
We have several thoughts in mind about what should our repo focus on.
- [ ] Add PP-nnunet with acceleration in preprocess, automatic configuration for all dataset and better performance compared to nnunet.
- [ ] Add top 1 liver segmentation algorithm on LITS challenge.
- [ ] Add 3D Vertebral Measurement System.
- [ ] Add pretrain model on various dataset.

## Acknowledgement
- Many thanks to [Lin Han](https://github.com/linhandev), [Lang Du](https://github.com/justld), [onecatcn](https://github.com/onecatcn) for their contribution in  our repository
- Many thanks to [itkwidgets](https://github.com/InsightSoftwareConsortium/itkwidgets) for their powerful visualization toolkit that we used to present our visualizations.

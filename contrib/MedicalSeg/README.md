English | [简体中文](README_CN.md)

# MedicalSeg
MedicalSeg is an easy-to-use 3D medical image segmentation toolkit that supports the whole segmentation process including data preprocessing, model training, and model deployment. Specially, We provide data preprocessing acceleration, high precision model on [COVID-19 CT scans](https://www.kaggle.com/andrewmvd/covid19-ct-scans) lung dataset and [MRISpineSeg](https://aistudio.baidu.com/aistudio/datasetdetail/81211) spine dataset, support for multiple datasets including [MSD](http://medicaldecathlon.com/), [Promise12](https://promise12.grand-challenge.org/), [Prostate_mri](https://liuquande.github.io/SAML/) and etc, and a [3D visualization demo](visualize.ipynb) based on [itkwidgets](https://github.com/InsightSoftwareConsortium/itkwidgets). The following image visualize the segmentation results on these two datasets:


<p align="center">
<img src="https://github.com/shiyutang/files/raw/main/ezgif.com-gif-maker%20(1).gif" width="30.6%" height="20%"><img src="https://github.com/shiyutang/files/raw/main/ezgif.com-gif-maker.gif" width="40.6%" height="20%">
<p align="center">
    VNet segmentation result on COVID-19 CT scans (mDice on evalset is 97.04%) & MRISpineSeg (16 class mDice on evalset is 89.14%)
</p>
</p>


**MedicalSeg is currently under development! If you find any problem using it or want to share any future develop suggestions, please open a github issue or join us by scanning the following wechat QR code.**

<p align="center">
<img src="https://user-images.githubusercontent.com/48433081/163670177-c59d21cd-44a8-4af2-acda-2b62c188cd3b.png" width="20%" height="20%">
</p>


## Contents
1. [Performance](##Performance)
2. [Quick Start](##QuickStart)
3. [Structure](#Structure)
4. [TODO](#TODO)
5. [Acknowledgement](#Acknowledgement)

## Performance

###  1. Accuracy

We successfully validate our framework with [Vnet](https://arxiv.org/abs/1606.04797) on the [COVID-19 CT scans](https://www.kaggle.com/andrewmvd/covid19-ct-scans) and [MRISpineSeg](https://www.spinesegmentation-challenge.com/) dataset. With the lung mask as label, we reached dice coefficient of 97.04% on COVID-19 CT scans. You can download the log to see the result or load the model and validate it by yourself :).

#### **Result on COVID-19 CT scans**

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
We add GPU acceleration in data preprocess using [CuPy](https://docs.cupy.dev/en/stable/index.html). Compared with preprocess data on CPU, acceleration enable us to use about 40% less time in data prepeocessing. The following shows the time we spend in process COVID-19 CT scans.

<center>

| Device | Time(s) |
|:-:|:-:|
|CPU|50.7|
|GPU|31.4( &#8595; 38%)|

</center>


## QuickStart
This part introduce a easy to use the demo on COVID-19 CT scans dataset. This demo is available on our [Aistudio project](https://aistudio.baidu.com/aistudio/projectdetail/3519594) as well. Detailed steps on training and add your own dataset can refer to this [tutorial](documentation/tutorial.md).
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

- Get and preprocess the data. Remember to replace prepare_lung_coronavirus.py with different python script that you need [here](./tools):
    - change the GPU setting [here](tools/preprocess_globals.yml) to True if you installed CuPY and want to use GPU to accelerate.
    ```
    python tools/prepare_lung_coronavirus.py
    ```

- Run the train and validation example. (Refer to the [tutorial](documentation/tutorial.md) for details.)
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
├── run-vnet.sh     # the script to reproduce our project, including training, validate, infer and deploy
├── tools           # Data preprocess including fetch data, process it and split into training and validation set
├── train.py
├── val.py
└── visualize.ipynb # You can try to visualize the result use this file.
```

## TODO
We have several thoughts in mind about what should our repo focus on. Your contribution will be very much welcomed.
- [ ] Add PP-nnunet with acceleration in preprocess, automatic configuration for all dataset and better performance compared to nnunet.
- [ ] Add top 1 liver segmentation algorithm on LITS challenge.
- [ ] Add 3D Vertebral Measurement System.
- [ ] Add pretrain model on various dataset.

## Acknowledgement
- Many thanks to [Lin Han](https://github.com/linhandev), [Lang Du](https://github.com/justld), [onecatcn](https://github.com/onecatcn) for their contribution in  our repository
- Many thanks to [itkwidgets](https://github.com/InsightSoftwareConsortium/itkwidgets) for their powerful visualization toolkit that we used to present our visualizations.

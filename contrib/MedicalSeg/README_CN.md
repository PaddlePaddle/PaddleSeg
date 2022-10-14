[English](README.md) | 简体中文

# MedicalSeg 介绍
MedicalSeg 是一个简单易使用的全流程 3D 医学图像分割工具包，它支持从数据预处理、训练评估、再到模型部署的全套分割流程。特别的，我们还提供了数据预处理加速，在肺部数据 [COVID-19 CT scans](https://www.kaggle.com/andrewmvd/covid19-ct-scans) 和椎骨数据 [MRISpineSeg](https://aistudio.baidu.com/aistudio/datasetdetail/81211) 上的高精度模型， 对于[MSD](http://medicaldecathlon.com/)、[Promise12](https://promise12.grand-challenge.org/)、[Prostate_mri](https://liuquande.github.io/SAML/)等数据集的支持，以及基于[itkwidgets](https://github.com/InsightSoftwareConsortium/itkwidgets) 的 3D 可视化[Demo](visualize.ipynb)。如图所示是基于 MedicalSeg 在 Vnet 上训练之后的可视化结果：

<p align="center">
<img src="https://github.com/shiyutang/files/raw/main/ezgif.com-gif-maker%20(1).gif" width="30.6%" height="20%"><img src="https://github.com/shiyutang/files/raw/main/ezgif.com-gif-maker.gif" width="40.6%" height="20%">
<p align="center">
    Vnet 在 COVID-19 CT scans (评估集上的 mDice 指标为 97.04%) 和 MRISpineSeg 数据集(评估集上的 16 类 mDice 指标为 89.14%) 上的分割结果
</p>
</p>

如果大家有使用问题和功能建议, 可以通过[GitHub Issues](https://github.com/PaddlePaddle/PaddleSeg/issues)提issue。
**欢迎大家加入PaddleSeg的微信用户群👫**（扫码填写问卷即可入群），和各界大佬交流学习，还可以**领取重磅大礼包🎁**
  * 🔥 获取PaddleSeg的历次直播视频，最新发版信息和直播动态
  * 🔥 获取PaddleSeg自建的人像分割数据集，整理的开源数据集
  * 🔥 获取PaddleSeg在垂类场景的预训练模型和应用合集，涵盖人像分割、交互式分割等等
  * 🔥 获取PaddleSeg的全流程产业实操范例，包括质检缺陷分割、抠图Matting、道路分割等等
<div align="center">
<img src="https://user-images.githubusercontent.com/48433081/174770518-e6b5319b-336f-45d9-9817-da12b1961fb1.jpg"  width = "200" />  
</div>

## Contents
1. [模型性能](##模型性能)
2. [快速开始](##快速开始)
3. [代码结构](#代码结构)
4. [TODO](#TODO)
5. [致谢](#致谢)

## 模型性能

###  1. 精度

我们使用 [Vnet](https://arxiv.org/abs/1606.04797) 在 [COVID-19 CT scans](https://www.kaggle.com/andrewmvd/covid19-ct-scans) 和 [MRISpineSeg](https://www.spinesegmentation-challenge.com/) 数据集上成功验证了我们的框架。以左肺/右肺为标签，我们在 COVID-19 CT scans 中达到了 97.04% 的 mDice 系数。你可以下载日志以查看结果或加载模型并自行验证:)。

#### **COVID-19 CT scans 上的分割结果**


| 骨干网络 | 分辨率 | 学习率 | 训练轮数 | mDice | 链接 |
|:-:|:-:|:-:|:-:|:-:|:-:|
|-|128x128x128|0.001|15000|97.04%|[model](https://bj.bcebos.com/paddleseg/paddleseg3d/lung_coronavirus/vnet_lung_coronavirus_128_128_128_15k_1e-3/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/paddleseg3d/lung_coronavirus/vnet_lung_coronavirus_128_128_128_15k_1e-3/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=9db5c1e11ebc82f9a470f01a9114bd3c)|
|-|128x128x128|0.0003|15000|92.70%|[model](https://bj.bcebos.com/paddleseg/paddleseg3d/lung_coronavirus/vnet_lung_coronavirus_128_128_128_15k_3e-4/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/paddleseg3d/lung_coronavirus/vnet_lung_coronavirus_128_128_128_15k_3e-4/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=0fb90ee5a6ea8821c0d61a6857ba4614)|

#### **MRISpineSeg 上的分割结果**


| 骨干网络 | 分辨率 | 学习率 | 训练轮数 | mDice(20 classes) | Dice(16 classes) | 链接 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|-|512x512x12|0.1|15000|74.41%| 88.17% |[model](https://bj.bcebos.com/paddleseg/paddleseg3d/mri_spine_seg/vnet_mri_spine_seg_512_512_12_15k_1e-1/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/paddleseg3d/mri_spine_seg/vnet_mri_spine_seg_512_512_12_15k_1e-1/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=36504064c740e28506f991815bd21cc7)|
|-|512x512x12|0.5|15000|74.69%| 89.14% |[model](https://bj.bcebos.com/paddleseg/paddleseg3d/mri_spine_seg/vnet_mri_spine_seg_512_512_12_15k_5e-1/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/paddleseg3d/mri_spine_seg/vnet_mri_spine_seg_512_512_12_15k_5e-1/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/index?id=08b0f9f62ebb255cdfc93fd6bd8f2c06)|


### 2. 速度
我们使用 [CuPy](https://docs.cupy.dev/en/stable/index.html) 在数据预处理中添加 GPU 加速。与 CPU 上的预处理数据相比，加速使我们在数据预处理中使用的时间减少了大约 40%。下面显示了加速前后，我们花在处理 COVID-19 CT scans 数据集预处理上的时间。

<center>

| 设备 | 时间(s) |
|:-:|:-:|
|CPU|50.7|
|GPU|31.4( &#8595; 38%)|

</center>


## 快速开始
这一部部分我们展示了一个快速在 COVID-19 CT scans 数据集上训练的例子，这个例子同样可以在我们的[Aistudio 项目](https://aistudio.baidu.com/aistudio/projectdetail/3519594)中找到。详细的训练部署，以及在自己数据集上训练的步骤可以参考这个[教程](documentation/tutorial_cn.md)。
- 下载仓库：
    ```
    git clone https://github.com/PaddlePaddle/PaddleSeg.git

    cd contrib/MedicalSeg/
    ```
- 安装需要的库：
    ```
    pip install -r requirements.txt
    ```
- (可选) 如果需要GPU加速，则可以参考[教程](https://docs.cupy.dev/en/latest/install.html) 安装 CuPY。

- 一键数据预处理。如果不是准备肺部数据，可以在这个[目录](./tools)下，替换你需要的其他数据：
    - 如果你安装了CuPY并且想要 GPU 加速，修改[这里](tools/preprocess_globals.yml)的 use_gpu 配置为 True。
    ```
    python tools/prepare_lung_coronavirus.py
    ```

- 基于脚本进行训练、评估、部署： (参考[教程](documentation/tutorial_cn.md)来了解详细的脚本内容。)
   ```
   sh run-vnet.sh
   ```

## 代码结构
这部分介绍了我们仓库的整体结构，这个结构决定了我们的不同的功能模块都是十分方便拓展的。我们的文件树如图所示：

```bash
├── configs         # 关于训练的配置，每个数据集的配置在一个文件夹中。基于数据和模型的配置都可以在这里修改
├── data            # 存储预处理前后的数据
├── deploy          # 部署相关的文档和脚本
├── medicalseg  
│   ├── core        # 训练和评估的代码
│   ├── datasets  
│   ├── models  
│   ├── transforms  # 在线变换的模块化代码
│   └── utils  
├── export.py
├── run-unet.sh     # 包含从训练到部署的脚本
├── tools           # 数据预处理文件夹，包含数据获取，预处理，以及数据集切分
├── train.py
├── val.py
└── visualize.ipynb # 用于进行 3D 可视化
```

## TODO
未来，我们想在这几个方面来发展 MedicalSeg，欢迎加入我们的开发者小组。
- [ ] 增加带有预训练加速，自动化参数配置的高精度 PP-nnunet 模型。
- [ ] 增加在 LITs 挑战中的 Top 1 肝脏分割算法。
- [ ] 增加 3D 椎骨可视化测量系统。
- [ ] 增加在多个数据上训练的预训练模型。


## 致谢
- 非常感谢 [Lin Han](https://github.com/linhandev), [Lang Du](https://github.com/justld), [onecatcn](https://github.com/onecatcn) 对我们仓库的贡献。
- 非常感谢 [itkwidgets](https://github.com/InsightSoftwareConsortium/itkwidgets) 强大的3D可视化功能。

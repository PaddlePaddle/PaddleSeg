[English](README.md) | 简体中文

# 医疗图像分割 MedicalSeg
医疗图像分割是对医疗成像生成的图像进行一个逐像素/体素的分类工作，从而能将不同的器官/组织进行区分，从而广泛用于医疗诊断和治疗规划中。医疗图像分割可分为2D医疗图像分割和3D医疗图像分割。2D的医疗图像分割可以直接使用PaddleSeg，详细可以见[眼底数据分割说明](../../configs/unet/)，3D图像的分割则由 MedicalSeg 负责。


MedicalSeg 是一个简单易使用的全流程 3D 医学图像分割工具包，它支持从数据预处理、训练评估、再到模型部署的全套分割流程。特别的，我们还提供了数据预处理加速，在肺部数据 [COVID-19 CT scans](https://www.kaggle.com/andrewmvd/covid19-ct-scans) 和椎骨数据 [MRISpineSeg](https://aistudio.baidu.com/aistudio/datasetdetail/81211) 上的高精度模型， 对于[MSD](http://medicaldecathlon.com/)、[Promise12](https://promise12.grand-challenge.org/)、[Prostate_mri](https://liuquande.github.io/SAML/)等数据集的支持，以及基于[itkwidgets](https://github.com/InsightSoftwareConsortium/itkwidgets) 的 3D 可视化[Demo](visualize.ipynb)。如图所示是基于 MedicalSeg 在 Vnet 上训练之后的可视化结果：

<p align="center">
<img src="https://github.com/shiyutang/files/raw/main/ezgif.com-gif-maker%20(1).gif" width="30.6%" height="20%"><img src="https://github.com/shiyutang/files/raw/main/ezgif.com-gif-maker.gif" width="40.6%" height="20%">
<p align="center">
    Vnet 在 COVID-19 CT scans (评估集上的 mDice 指标为 97.04%) 和 MRISpineSeg 数据集(评估集上的 16 类 mDice 指标为 89.14%) 上的分割结果
</p>
</p>


<p align="center">
<img src="https://github.com/shiyutang/files/raw/main/meeicalsegall.png" width="70.6%" height="20%">
<p align="center">
    <b>MedicalSeg 全景图</b>
</p>
</p>

**MedicalSeg 刚刚发布全新版本！如果您在使用中发现任何问题，或想分享任何开发建议，请提交 github issue 或扫描以下微信二维码加入我们。**

<p align="center">
<img src="https://github.com/shiyutang/files/raw/main/medicalsegwechat.jpg" width="25%" height="20%">
</p>

## 目录结构
0. [最新消息](##最新消息)
1. [3D智能标注EISeg-Med3D](##3D智能标注EISeg-Med3D)
2. [MedicalSeg模型性能](##MedicalSeg模型性能)
3. [快速在肺部数据上开始](##快速在肺部数据上开始)
4. [完整教程](##完整教程)
5. [在自己的数据上训练](##在自己的数据上训练)
6. [代码结构](##代码结构)
7. [TODO](##TODO)
8. [License](##License)
9. [致谢](##致谢)


## <img src="https://user-images.githubusercontent.com/34859558/190043516-eed25535-10e8-4853-8601-6bcf7ff58197.png" width="25"/> 最新消息
- [2022-9] 新增3大前沿模型支持全流程部署应用，包含 nnformer, TransUnet 和 nnUnet，让你体会更强更精准的分割效果；新增 3D 医疗图像智能标注平台 [EISeg-Med3D](../../EISeg/med3d/README.md)，方便快捷地实现精准3D医疗图像标注。
- [2022-4] MedicalSeg 发布0.1版本，提供了3D医疗图像分割中的数据预处理到到训练部署全流程，包含了对五个数据集的原生支持，以及椎骨和肺部上的高精度预训练模型。

## <img src="https://user-images.githubusercontent.com/34859558/188419267-bd117697-7456-4c72-8cbe-1272264d4fe4.png" width="25"/> 3D智能标注EISeg-Med3D
为了解决3D医疗手工标注效率低下的问题，并从数据标注开始真正全流程用AI赋能医疗，我们基于医疗软件 Slicer 搭建了[EISeg-Med3D](../../EISeg/med3d/README.md)，一个专注用户友好、高效、智能的3D医疗图像标注平台，通过在标注过程中融入3D交互式分割模型实现3D医疗数据标注的智能化高效化。主要特性如下：

* **高效**：每个类别只需**数次点击**直接生成3d分割结果，从此告别费时费力的手工标注。

* **准确**：点击 3 点 mIOU 即可达到0.85，配合搭载机器学习算法和手动标注的标注编辑器，精度 100% 不是梦。

* **便捷**：三步轻松安装；标注结果、进度自动保存；标注结果透明度调整提升标注准确度；用户友好的界面交互，让你标注省心不麻烦。

<div align="center">
<p align="center">
  <img src="https://user-images.githubusercontent.com/34859558/188415269-10526530-0415-4632-8223-0e5d755db29c.gif"  align="middle" width = 600"/>
</p>
</div>

------------------


## <img src="https://user-images.githubusercontent.com/34859558/190044217-8f6befc2-7f20-473d-b356-148e06265205.png" width="25"/> MedicalSeg模型性能

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

## <img src="https://user-images.githubusercontent.com/34859558/190043857-bfbdaf8b-d2dc-4fff-81c7-e0aac50851f9.png" width="25"/> 快速在肺部数据上开始
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

## <img src="https://user-images.githubusercontent.com/34859558/188439970-18e51958-61bf-4b43-a73c-a9de3eb4fc79.png" width="25"/> 完整教程
这一部分，我们将会介绍参数配置、训练、评估、部署部分的完整内容。


### 1. 参数配置
配置文件的结构如下所示：
```bash
├── _base_                   # 一级基础配置，后面所有的二级配置都需要继承它，你可以在这里设置自定义的数据路径，确保它有足够的空间来存储数据。
│   └── global_configs.yml
├── lung_coronavirus         # 每个数据集/器官有个独立的文件夹，这里是 COVID-19 CT scans 数据集的路径。
│   ├── lung_coronavirus.yml # 二级配置，继承一级配置，关于损失、数据、优化器等配置在这里。
│   ├── README.md  
│   └── vnet_lung_coronavirus_128_128_128_15k.yml    # 三级配置，关于模型的配置，不同的模型可以轻松拥有相同的二级配置。
└── schedulers              # 用于规划两阶段的配置，暂时还没有使用它。
    └── two_stage_coarseseg_fineseg.yml
```


### 2. 数据准备
我们使用数据准备脚本来进行一键自动化的数据下载、预处理变换、和数据集切分。只需要运行下面的脚本就可以一键准备好数据：
```
python tools/prepare_lung_coronavirus.py  # 以 CONVID-19 CT scans 为例。
```

### 3. 训练、评估
准备好配置之后，只需要一键运行 [run-vnet.sh](../run-vnet.sh) 就可以进行训练和评估。让我们看看这个脚本中的命令是什么样子的：

```bash
# 设置使用的单卡 GPU id
export CUDA_VISIBLE_DEVICES=0

# 设置配置文件名称和保存路径
yml=vnet_lung_coronavirus_128_128_128_15k
save_dir=saved_model/${yml}
mkdir save_dir

# 训练模型
python3 train.py --config configs/lung_coronavirus/${yml}.yml \
--save_dir  $save_dir \
--save_interval 500 --log_iters 100 \
--num_workers 6 --do_eval --use_vdl \
--keep_checkpoint_max 5  --seed 0  >> $save_dir/train.log

# 评估模型
python3 val.py --config configs/lung_coronavirus/${yml}.yml \
--save_dir  $save_dir/best_model --model_path $save_dir/best_model/model.pdparams

```


### 4. 模型部署
得到训练好的模型之后，我们可以将它导出为静态图来进行推理加速，下面的步骤就可以进行导出和部署，详细的教程则可以参考[这里](../deploy/python/README.md)：

```bash
cd MedicalSeg/

# 用训练好的模型进行静态图导出
python export.py --config configs/lung_coronavirus/vnet_lung_coronavirus_128_128_128_15k.yml --model_path /path/to/your/trained/model

# 使用 Paddle Inference 进行推理
python deploy/python/infer.py \
    --config /path/to/model/deploy.yaml \
    --image_path /path/to/image/path/or/dir/
    --benchmark True   # 在安装了 AutoLog 之后，打开benchmark可以看到推理速度等信息，安装方法可以见 ../deploy/python/README.md

```
如果有“Finish” 输出，说明导出成功，并且可以进行推理加速。

## <img src="https://user-images.githubusercontent.com/34859558/190044556-ad04dc0e-3ec9-41c4-b6a5-a3d251f5cad2.png" width="25"/> 在自己的数据上训练
如果你想要定制化地针对自己的数据进行训练，你需要增加一个[数据集代码](../medicalseg/datasets/lung_coronavirus.py), 一个 [数据预处理代码](../tools/prepare_lung_coronavirus.py), 一个和这个数据集相关的[配置目录](../configs/lung_coronavirus), 一份 [训练脚本](../run-vnet.sh)。这些修改只需要依照已有代码进行依葫芦画瓢即可，下面我们分步骤来看这些部分都需要增加什么：

### 1 增加配置目录
首先，我们如下图所示，增加一个和你的数据集相关的配置目录：
```
├── _base_
│   └── global_configs.yml
├── lung_coronavirus
│   ├── lung_coronavirus.yml
│   ├── README.md
│   └── vnet_lung_coronavirus_128_128_128_15k.yml
```

### 2 增加数据集预处理文件
所有数据需要经过预处理转换成 numpy 数据并进行数据集划分，参考这个[数据预处理代码](../tools/prepare_lung_coronavirus.py)：
```python
├── lung_coronavirus_phase0  # 预处理后的文件路径
│   ├── images
│   │   ├── imagexx.npy
│   │   ├── ...
│   ├── labels
│   │   ├── labelxx.npy
│   │   ├── ...
│   ├── train_list.txt       # 训练数据，格式:  /path/to/img_name_xxx.npy /path/to/label_names_xxx.npy
│   └── val_list.txt         # 评估数据，格式:  img_name_xxx.npy label_names_xxx.npy
```

### 3 增加数据集文件
所有的数据集都继承了 MedicalDataset 基类，并通过上一步生成的 train_list.txt 和 val_list.txt 来获取数据。代码示例在[这里](../medicalseg/datasets/lung_coronavirus.py)。

### 4 增加训练脚本
训练脚本能自动化训练推理过程，我们提供了一个[训练脚本示例](../run-vnet.sh) 用于参考，只需要复制，并按照需要修改就可以进行一键训练推理：
```bash
# 设置使用的单卡 GPU id
export CUDA_VISIBLE_DEVICES=3

# 设置配置文件名称和保存路径
config_name=vnet_lung_coronavirus_128_128_128_15k
yml=lung_coronavirus/${config_name}
save_dir_all=saved_model
save_dir=saved_model/${config_name}
mkdir -p $save_dir

# 模型训练
python3 train.py --config configs/${yml}.yml \
--save_dir  $save_dir \
--save_interval 500 --log_iters 100 \
--num_workers 6 --do_eval --use_vdl \
--keep_checkpoint_max 5  --seed 0  >> $save_dir/train.log

# 模型评估
python3 val.py --config configs/${yml}.yml \
--save_dir  $save_dir/best_model --model_path $save_dir/best_model/model.pdparams \

# 模型导出
python export.py --config configs/${yml}.yml \
--model_path $save_dir/best_model/model.pdparams

# 模型预测
python deploy/python/infer.py  --config output/deploy.yaml --image_path data/lung_coronavirus/lung_coronavirus_phase0/images/coronacases_org_007.npy  --benchmark True

```


## <img src="https://user-images.githubusercontent.com/34859558/190046287-31b0467c-1a7e-4bf2-9e5e-40ff3eed94ee.png" width="25"/> 代码结构
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


## <img src="https://user-images.githubusercontent.com/34859558/190046674-53e22678-7345-4bf1-ac0c-0cc99718b3dd.png" width="25"/> TODO
未来，我们想在这几个方面来发展 MedicalSeg，欢迎加入我们的开发者小组。
- [✔️] 增加带有预训练加速，自动化参数配置的高精度 PP-nnunet 模型。
- [✔️] 增加 3D 智能标注平台。
- [ ] 增加在多个数据上训练的预训练模型。

## <img src="https://user-images.githubusercontent.com/34859558/188446853-6e32659e-8939-4e65-9282-68909a38edd7.png" width="25"/> License

MedicalSeg 的 License 为 [Apache 2.0 license](LICENSE).

## <img src="https://user-images.githubusercontent.com/34859558/188446803-06c54d50-f2aa-4a53-8e08-db2253df52fd.png" width="25"/> 致谢
- 非常感谢 [Lin Han](https://github.com/linhandev), [Lang Du](https://github.com/justld), [onecatcn](https://github.com/onecatcn) 对我们仓库的贡献。
- 非常感谢 [itkwidgets](https://github.com/InsightSoftwareConsortium/itkwidgets) 强大的3D可视化功能。
- 非常感谢 <a href="https://www.flaticon.com/free-icons/idea" title="idea icons"> Idea icons created by Vectors Market - Flaticon</a> 给我们提供了好看的图标

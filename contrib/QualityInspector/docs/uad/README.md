# 无监督异常检测算法

目前uad工具支持三种前沿的无监督异常检测算法，分别是[PaDiM](../../configs/uad/padim/README.md), [PatchCore](../../configs/uad/patchcore/README.md)以及[STFPM](../../configs/uad/stfpm/README.md)，其中，PaDiM、PatchCore无需训练网络，未来将支持更多无监督异常检测算法。


## 1. 数据集

此工具以MVTec AD数据集为例, 首先下载数据集[下载MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/), 数据保存在`data/mvtec_anomaly_detection/{category}/...`目录中。MVTec AD数据包含结构和纹理类型的零件共计15类，其中训练集只包含OK图像，测试集包含NG和OK图像，示意图如下：

![](https://github.com/Sunting78/images/blob/master/mvtec.png)

另外，如果希望使用自己的数据集, 请组织成MVTec AD数据集的格式, 将自定义数据集作为MVTec AD中的一个category，路径设置为`{repo}/data/mvtec_anomaly_detection/{category}/...`，标签文件为灰度图, 缺陷部分像素值为255;


## 2. 训练、评估、预测命令

以PaDiM模型为例，训练、评估、预测脚本存放在`tools/uad/padim/`目录下，通过`--config`参数传入对应模型YML配置文件，通过`--category`参数指定MVTec AD中的某个类别:

* 训练:

```python tools/uad/padim/train.py --config ./configs/uad/padim/padim_resnet18_mvtec.yml --category bottle```

* 评估:

```python tools/uad/padim/val.py --config ./configs/uad/padim/padim_resnet18_mvtec.yml --category bottle```

* 预测:

```python tools/uad/padim/predict.py --config ./configs/uad/padim/padim_resnet18_mvtec.yml --category bottle```



## 3. 配置文件解读

无监督异常检测(uad)模型的参数可以通过YML配置文件和命令行参数两种方式指定, 如果YML文件与命令行同时指定一个参数, 命令行指定的优先级更高, 以PaDiM的YML文件为例, 主要包含以下参数:

```
# common arguments
device: gpu
seed: 3   # 指定numpy, paddle的随机种子

# dataset arguments
batch_size: 1
category: bottle # 指定MVTecAD数据集的某个类别
resize: [256, 256]  # 指定读取图像的resize尺寸
crop_size: [224, 224]  # 指定resize图像的crop尺寸
data_path: data/mvtec_anomaly_detection  # 指定MVTecAD数据集的根目录
save_path: output/    # 指定训练时模型参数保存的路径和评估/预测时结果图片的保存路径

# train arguments
do_eval: True  # 指定训练后是否进行评估
backbone: resnet18  # 支持resnet18, resnet50, wide_resnet50_2

# val and predict arguments
save_pic: True   # 指定是否保存第一张评估图片/预测图片的结果
model_path: output/resnet18/bottle/bottle.pdparams # 指定加载模型参数的路径

# predict arguments
img_path: data/mvtec_anomaly_detection/bottle/test/broken_large/000.png  # 指定预测的图片路径
threshold: 0.5    # 指定预测后二值化异常分数图的阈值
```

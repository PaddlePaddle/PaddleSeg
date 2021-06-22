# 训练iann可用的自定义模型

目前已经可以通过简单的配置完成模型训练了，但其中有些设置还不能通过配置文件进行修改。

## 一、数据组织

在需要训练自己的数据集时，目前需要将数据集构造为如下格式，直接放在datasets文件夹中。文件名可以根据要求来进行设置，只需要在配置文件中设定好即可，图像和标签与平时使用的分割图像的用法相同。

```
datasets
    |
    ├── train_data
    |       ├── img
    |       |    └── filename_1.jpg
    |       └── gt
    |            └── filename_1.png
    |
    └── eval_data
            ├── img
            |    └── filename_1.jpg
            └── gt
                 └── filename_1.png
```

## 二、训练

直接运行ritm_train.py即可开始训练。

```python
%cd train
! python ritm_train.py --config train_config.yaml
```

目前一些简单的参数已经可以在yaml配置文件中进行自定义设置，不过现阶段仍然不够灵活，可能出现各种问题。

```
iters: 100000  # 训练轮数
batch_size: 16  # bs大小
save_interval: 1000  # 保存间隔
log_iters: 10  # 打印log的间隔
worker: 4  # 子进程数
save_dir: model_output  # 保存路径
use_vdl: False  # 是否使用vdl

dataset:
  dataset_path: iann/train/datasets  # 数据集所在路径
  image_name: img  # 图像文件夹的名称
  label_name: gt  # 标签文件夹的名称

train_dataset:  # 训练数据
  crop_size: [320, 480]  # 裁剪大小
  folder_name: train_data  # 训练数据文件夹的名称

val_dataset:  # 验证数据
  folder_name: val_data  # 验证数据文件夹的名称

optimizer:
  type: adam  # 优化器，目前仅可以选择‘adam’和‘sgd’

learning_rate:
  value_1: 5e-5  # 需要设置两个学习率
  value_2: 5e-6
  decay:
    type: poly  # 学习率衰减，目前仅支持‘poly’，可以修改下面的参数
    steps: 1000
    power: 0.9
    end_lr: 0.0

model:
  type: deeplab  # 模型名称，目前支持‘hrnet’、‘deeplab’以及‘shufflenet’
  backbone: resnet18  # 下面的参数是模型对应的参数，可在源码中查看
  is_ritm: True
  weights: None  # 加载权重的路径
```



### * 说明

1. 这里有个坑，数据不能有没有标签的纯背景，这样找不到正样点训练就会卡住，并且还不报错。


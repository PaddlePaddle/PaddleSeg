本目录下的配置文件用于飞桨模型选型工具（Paddle-SMRT）。
# 简介

我们以一个示例讲解分割模型的训练、部署。

# 准备
## 准备环境

参考PaddleSeg[安装文档](../../docs/install_cn.md)安装PaddlePaddle、下载PaddleSeg代码、安装PaddleSeg依赖库。

## 准备数据

我们准备了一个缺陷分割的数据集，点击[链接](https://paddle-smrt.bj.bcebos.com/data/seg/defect_data.zip)下载或者执行如下命令下载。
```
wget https://paddle-smrt.bj.bcebos.com/data/seg/defect_data.zip
```

将下载的数据集解压，并存放到`PaddleSeg/data/defect_data`目录，如下。
```
PaddleSeg/data/defect_data
├── Annotations
├── JPEGImages
├── test.txt
├── train.txt
└── val.txt
```

该数据集的原始图片保存在JPEGImages文件，标注图片保存在Annotations。

train.txt、val.txt和test.txt分别标识用于训练、验证和测试的数据，其中内容如下。
```
JPEGImages/liewen_26.png Annotations/liewen_26.png
JPEGImages/diaojiao_394.png Annotations/diaojiao_394.png
```

该数据集包含背景和3类缺陷目标，总共4类，分别标注的数值使0，1，2，3。

详细的数据准备方法，请参考[数据标注文档](../../docs/data/marker/marker_cn.md)和[数据配置文档](../../docs/data/custom/data_prepare_cn.md)。

# 模型训练


```
export CUDA_VISIBLE_DEVICES=0 # 设置1张可用的卡
# windows下请执行以下命令
# set CUDA_VISIBLE_DEVICES=0
python train.py \
       --config configs/smrt/pp_liteseg_stdc2.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output
```

详细的模型训练方法，请参考[文档](docs/train/train_cn.md)。


# 模型部署

# 图像分割：从 PaddleLabel 到 PaddleSeg

PaddleLabel 标注数据+PaddleSeg 训练预测=快速完成一次图像分割的任务

---

## 1. 数据准备

- 首先使用`PaddleLabel`对自制的狗子数据集进行标注，其次使用`Split Dataset`功能分割数据集，最后导出数据集
- 从`PaddleLabel`导出后的内容全部放到自己的建立的文件夹下，例如`dog_seg_dataset`，其目录结构如下：

```
├── dog_seg_dataset
│   ├── Annotations
│   ├── JPEGImages
│   ├── labels.txt
│   ├── test_list.txt
│   ├── train_list.txt
│   ├── val_list.txt
```

## 2. 训练

### 2.1 安装必备的库

**2.1.1 安装 paddlepaddle**

```
# 您的机器安装的是 CUDA9 或 CUDA10，请运行以下命令安装
pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
# 您的机器是CPU，请运行以下命令安装
# pip install paddlepaddle
```

**2.1.2 安装 paddleseg 以及依赖项**

```
git clone https://gitee.com/paddlepaddle/PaddleSeg.git
cd PaddleSeg
pip install -r requirements.txt
python setup.py install
```

### 2.2 准备自制的狗子分割数据集

```
cd ./PaddleSeg/data/
mkdir dog_seg_dataset
cd ../../
cp -r ./dog_seg_dataset/* ./PaddleSeg/data/dog_seg_dataset
```

### 2.3 修改配置文件

这里改了`FCN`模型的配置文件，改完后拷贝了一份放置到了`configs`目录下并重命名为`mynet.yml`，关于自定义数据集的配置可以参考`PaddleSeg`在`GitHub`上的说明[配置文件说明](https://gitee.com/paddlepaddle/PaddleSeg/blob/release/2.4/docs/design/use/use_cn.md)

> PaddleSeg/configs/mynet.yml

```
batch_size: 4 # 迭代一次送入网络的图片数量
iters: 10000 # 模型迭代次数

train_dataset:
  type: Dataset # 数据集格式，自定义数据集用Dataset
  dataset_root: ../data/dog_seg_dataset # 训练数据集存放的目录
  train_path: ../data/dog_seg_dataset/train_list.txt
  num_classes: 2 # 像素类别数（背景也算为一类
  transforms: # 数据变换与数据增强
    - type: ResizeStepScaling # 对图像按照某一个比例进行缩放，这个比例以scale_step_size为步长
      min_scale_factor: 0.5 # 缩放过程中涉及的参数
      max_scale_factor: 2.0
      scale_step_size: 0.25
    - type: RandomPaddingCrop # 对图像和标注图进行随机裁剪
      crop_size: [512, 512]
    - type: RandomHorizontalFlip # 以一定的概率对图像进行水平翻转
    - type: RandomDistort # 对图像像素进行处理
      brightness_range: 0.4
      contrast_range: 0.4
      saturation_range: 0.4
    - type: Normalize # 对图像进行标准化
  mode: train # 训练模式

val_dataset:
  type: Dataset
  dataset_root: ../data/dog_seg_dataset # 验证数据集存放的目录
  val_path: ../data/dog_seg_dataset/val_list.txt
  transforms:
    - type: Normalize
  mode: val # 验证模式
  num_classes: 2

optimizer: # 设定优化器的类型
  type: sgd # 随机梯度下降
  momentum: 0.9 # 动量
  weight_decay: 0.0005 # 权值衰减，使用的目的是防止过拟合

lr_scheduler: # 学习率的相关设置
  type: PolynomialDecay # 一种学习率类型。共支持12种策略
  learning_rate: 0.01
  power: 0.9
  end_lr: 0

loss: # 损失函数设置
  types:
    - type: CrossEntropyLoss # 交叉熵损失函数
  coef: [1] # 当使用了多种损失函数，可在 coef 中为每种损失指定配比

model: # 使用何种语义分割模型
  type: FCN
  backbone: # 使用何种骨干网络
    type: HRNet_W48
    pretrained: https://bj.bcebos.com/paddleseg/dygraph/hrnet_w48_ssld.tar.gz # 预训练模型
  num_classes: 2
  pretrained: Null
  backbone_indices: [-1]
```

### 2.4 开始训练

```
export CUDA_VISIBLE_DEVICES=0

# 开始训练
# 边训练边测试
# --config 参数表示指定使用哪个配置文件
# --do_eval 参数表示一遍训练一遍验证
# --save_interval 参数表示每经过100个iters，进行一个模型的保存
python PaddleSeg/tools/train.py \
       --config PaddleSeg/configs/mynet.yml \
       --do_eval \
       --use_vdl \
       --save_interval 100 \
       --save_dir PaddleSeg/output
```

## 3. 模型评估

### 3.1 评估

```
# 评估 默认使用训练过程中保存的model_final.pdparams
python PaddleSeg/val.py \
       --config PaddleSeg/configs/mynet.yml \
       --model_path PaddleSeg/output/best_model/model.pdparams
```

### 3.2 预测

```
# image_path参数表示选择预测的图片
# save_dir参数表示预测保存的结果地址
python PaddleSeg/predict.py \
       --config PaddleSeg/configs/mynet.yml \
       --model_path PaddleSeg/output/best_model/model.pdparams \
       --image_path PaddleSeg/data/dog_seg_dataset/JPEGImages/e619b17a9c1b9f085dc2712eb603171f.jpeg \
       --save_dir PaddleSeg/output/result
```

预测的原图是`data/dog_seg_dataset/JPEGImages/e619b17a9c1b9f085dc2712eb603171f.jpeg`

<img src="https://ai-studio-static-online.cdn.bcebos.com/f9efa53cf0334146a0963d6033c2cb84c3540525b565454199f2a859f86b501e" width="50%" height="50%">

预测的结果`PaddleSeg/output/result`目录里面，如下图所示：

<img src="https://ai-studio-static-online.cdn.bcebos.com/8d6dea0d5fa24912a58612839026b255652d7d3ccf0a40aaa5e6056750f8f75b" width="50%" height="50%">

<img src="https://ai-studio-static-online.cdn.bcebos.com/6dfc7c24edda4a489b7f1629957be44be44d3b9c94d14becb88aa22e42a41d50" width="50%" height="50%">

## AI Studio 第三方教程推荐

[快速体验演示案例](https://aistudio.baidu.com/aistudio/projectdetail/4353528)

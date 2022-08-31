# 花朵分类：从 PaddleLabel 到 PaddleClas

PaddleLabel 标注数据+PaddleClas 训练预测=快速完成一次花朵分类的任务

---

## 1. 数据准备

- 首先使用`PaddleLabel`对自制的花朵数据集进行标注，其次使用`Split Dataset`功能分割数据集，最后导出数据集
- 从`PaddleLabel`导出后的内容全部放到自己的建立的文件夹下，例如`flower_clas_dataset`，其目录结构如下：

```
├── flower_clas_dataset
│   ├── image
│   │   ├── flower1.jpg
│   │   ├── flower2.jpg
│   │   ├── ...
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
# pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
# 您的机器是CPU，请运行以下命令安装
pip install paddlepaddle
```

**2.1.2 安装 paddleclas 以及依赖项**

```
git clone https://gitee.com/paddlepaddle/PaddleClas.git -b release/2.2
cd PaddleClas
pip install -r requirements.txt
python setup.py install
```

### 2.2 准备自制的花朵分类数据集

```
cd ./PaddleClas/dataset/
mkdir flower_clas_dataset
cd ../../
cp -r ./flower_clas_dataset/* ./PaddleClas/dataset/flower_clas_dataset
```

### 2.3 修改配置文件

> PaddleClas/ppcls/configs/quick_start/new_user/ShuffleNetV2_x0_25.yaml

```
# global configs
Global:
  checkpoints: null
  pretrained_model: null
  output_dir: ./output/
  device: cpu
  save_interval: 20
  eval_during_train: True
  eval_interval: 10
  epochs: 100
  print_batch_step: 10
  use_visualdl: True
  # used for static mode and model export
  image_shape: [3, 224, 224]
  save_inference_dir: ./inference

# model architecture
Arch:
  name: ShuffleNetV2_x0_25
  class_num: 3

# loss function config for traing/eval process
Loss:
  Train:
    - CELoss:
        weight: 1.0
  Eval:
    - CELoss:
        weight: 1.0


Optimizer:
  name: Momentum
  momentum: 0.9
  lr:
    name: Cosine
    learning_rate: 0.0125
    warmup_epoch: 5
  regularizer:
    name: 'L2'
    coeff: 0.00001


# data loader for train and eval
DataLoader:
  Train:
    dataset:
      name: ImageNetDataset
      image_root: ./dataset/
      cls_label_path: ./dataset/train_list.txt
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - RandCropImage:
            size: 224
        - RandFlipImage:
            flip_code: 1
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''

    sampler:
      name: DistributedBatchSampler
      batch_size: 16
      drop_last: False
      shuffle: True
    loader:
      num_workers: 0
      use_shared_memory: True

  Eval:
    dataset:
      name: ImageNetDataset
      image_root: ./dataset/
      cls_label_path: ./dataset/val_list.txt
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - ResizeImage:
            resize_short: 256
        - CropImage:
            size: 224
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
    sampler:
      name: DistributedBatchSampler
      batch_size: 32
      drop_last: False
      shuffle: False
    loader:
      num_workers: 0
      use_shared_memory: True

Infer:
  infer_imgs: dataset/predict_demo.jpg
  batch_size: 10
  transforms:
    - DecodeImage:
        to_rgb: True
        channel_first: False
    - ResizeImage:
        resize_short: 256
    - CropImage:
        size: 224
    - NormalizeImage:
        scale: 1.0/255.0
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
        order: ''
    - ToCHWImage:
  PostProcess:
    name: Topk
    topk: 3

Metric:
  Train:
    - TopkAcc:
        topk: [1, 3]
  Eval:
    - TopkAcc:
        topk: [1, 3]
```

### 2.4 添加类别映射文件

> PaddleClas/ppcls/configs/quick_start/new_user/label.txt

```
sunflower
rose
dandelion
```

### 2.5 开始训练

```
export CUDA_VISIBLE_DEVICES=0
# 开始训练
python PaddleClas/tools/train.py -c ./PaddleClas/ppcls/configs/quick_start/new_user/ShuffleNetV2_x0_25.yaml
```

## 3. 模型评估

### 3.1 评估

```
python PaddleClas/tools/eval.py -c ./PaddleClas/ppcls/configs/quick_start/new_user/ShuffleNetV2_x0_25.yaml
```

### 3.2 预测

```
python3 PaddleClas/tools/infer.py \
    -c ./PaddleClas/ppcls/configs/quick_start/new_user/ShuffleNetV2_x0_25.yaml \
    -o Infer.infer_imgs=dataset/predict_demo.jpg \
    -o Global.pretrained_model=output/ShuffleNetV2_x0_25/latest
```

预测的样例图片是：

<img src="https://ai-studio-static-online.cdn.bcebos.com/e7d6cabc46434205891cfc0c125b8dcec511e622469c49a5b8ec48051f7dd997" width="50%" height="50%">

预测的结果是：

> {'class_ids': [0, 1, 2], 'scores': [0.89812, 0.09476, 0.00712], 'file_name': 'dataset/predict_demo.jpg', 'label_names': []}
> 也就是说 0 的概率最大，为 0.89812，0 对应的结果是向日葵，也就是说结果是向日葵，预测无误。

## AI Studio 第三方教程推荐

[快速体验演示案例](https://aistudio.baidu.com/aistudio/projectdetail/4337003)

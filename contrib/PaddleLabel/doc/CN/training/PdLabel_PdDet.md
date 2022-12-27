# 道路标志检测：从 PaddleLabel 到 PaddleDetection

PaddleLabel 标注数据+PaddleDetection 训练预测=快速完成一次目标检测的任务

______________________________________________________________________

## 1. 数据准备

- 首先使用`PaddleLabel`对自制的路标数据集进行标注，其次使用`Split Dataset`功能分割数据集，最后导出数据集
- 从`PaddleLabel`导出后的内容全部放到自己的建立的文件夹下，例如`roadsign_det_dataset`，其目录结构如下：

```
├── roadsign_det_dataset
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

**2.1.2 安装 paddledetection 以及依赖项**

```
git clone https://gitee.com/paddlepaddle/PaddleDetection
cd PaddleDetection
pip install -r requirements.txt
python setup.py install
```

### 2.2 准备自制的路标数据集

```
cd ./PaddleDection/dataset/
mkdir roadsign_det_dataset
cd ../../
cp -r ./roadsign_det_dataset/* ./PaddleDection/dataset/roadsign_det_dataset
```

### 2.3 修改配置文件

该任务主要涉及 6 个配置文件，分别是:

1. 主配置文件入口：yolov3_mobilenet_v1_roadsign.yml
1. 定义训练数据的路径：roadsign_voc.yml
1. 定义公共参数：runtime.yml
1. 定义优化器的策略：optimizer_40e.yml
1. 定义模型和主干网络：yolov3_mobilenet_v1.yml
1. 定义数据预处理方式：yolov3_reader.yml

这里我们只需要改动一个配置文件即定义训练数据的路径的配置文件：

> PaddleDetection/configs/datasets/roadsign_voc.yml

```
metric: VOC
map_type: integral
num_classes: 4

TrainDataset:
  !VOCDataSet
    dataset_dir: dataset/roadsign_det_dataset
    anno_path: train_list.txt
    label_list: labels.txt
    data_fields: ['image', 'gt_bbox', 'gt_class', 'difficult']

EvalDataset:
  !VOCDataSet
    dataset_dir: dataset/roadsign_det_dataset
    anno_path: val_list.txt
    label_list: labels.txt
    data_fields: ['image', 'gt_bbox', 'gt_class', 'difficult']

TestDataset:
  !ImageFolder
    anno_path: dataset/roadsign_det_dataset/labels.txt
```

### 2.4 开始训练

```
export CUDA_VISIBLE_DEVICES=0
# 开始训练
python PeddleDetection/tools/train.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml --eval -o use_gpu=true
```

## 3. 模型评估

### 3.1 评估

```
python PeddleDetection/tools/eval.py -c PeddleDetection/configs/yolov3/yolov3_mobilenet_v1_roadsign.yml -o use_gpu=true
```

### 3.2 预测

```
python PeddleDetection/tools/infer.py \
    -c PeddleDetection/configs/yolov3/yolov3_mobilenet_v1_roadsign.yml \
    -o use_gpu=true \
    --infer_img=demo/road554.png
```

预测的样例图片如下图：

<img src="https://ai-studio-static-online.cdn.bcebos.com/8fb35c64f3424a098858a3f75255f0d56c6f9c9d7e24438c8d1bc2cd71e838d4" width="50%" height="50%">

预测的结果是：

> speedlimit 0.77 预测正确 ✔

## AI Studio 第三方教程推荐

[快速体验演示案例](https://aistudio.baidu.com/aistudio/projectdetail/4349280)

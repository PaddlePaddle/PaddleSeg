# SpatialEmbeddings

## 模型概述
本模型是基于proposal-free的实例分割模型，快速实时，同时准确率高，适用于自动驾驶等实时场景。

本模型基于KITTI中MOTS数据集训练得到，是论文 Segment as Points for Efficient Online Multi-Object Tracking and Segmentation中的分割部分
[论文地址](https://arxiv.org/pdf/2007.01550.pdf)

## KITTI MOTS指标
KITTI MOTS验证集AP:0.76, AP_50%:0.915

## 代码使用说明

### 1. 模型下载

执行以下命令下载并解压SpatialEmbeddings预测模型：

```
python download_SpatialEmbeddings_kitti.py
```

或点击[链接](https://paddleseg.bj.bcebos.com/models/SpatialEmbeddings_kitti.tar)进行手动下载并解压。

### 2. 数据下载

前往KITTI官网下载MOTS比赛数据[链接](https://www.vision.rwth-aachen.de/page/mots)

下载后解压到./data文件夹下, 并生成验证集图片路径的test.txt

### 3. 快速预测

使用GPU预测
```
python -u infer.py --use_gpu
```

使用CPU预测：
```
python -u infer.py
```
数据及模型路径等详细配置见config.py文件

#### 4. 预测结果示例：

  原图：

  ![](imgs/kitti_0007_000518_ori.png)

  预测结果：

  ![](imgs/kitti_0007_000518_pred.png)



## 引用

**论文**

*Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth*

**代码**

https://github.com/davyneven/SpatialEmbeddings

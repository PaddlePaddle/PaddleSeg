# STFPM
此模型是使用Paddle复现论文: [Student-Teacher Feature Pyramid Matching for Anomaly Detection](https://arxiv.org/abs/2103.04257v3).

![image](https://user-images.githubusercontent.com/61047836/227186808-0d58952e-1f1d-44ce-89f1-22a81b8e598d.png)

STFPM是一种基于表示的无监督异常检测算法，由预训练的教师网络和结构相同的学生网络组成。在训练时，学生网络通过将自身特征与教师网络中的对应特征相匹配来学习正样本图像的分布；在推理时，通过比较教师网络和学生网络的各个尺度特征，差异越大，异常分数越高。

## MVTec AD数据集上的实验结果

* 图像级和像素级ROCAUC指标:


|                       |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
|-----------------------|:-----:| :----: | :---: |:-------:|:-----:|:-----:|:------:|:-----:|:-------:|:--------:|:---------:|:-----:|:-----:|:----------:|:----------:|:------:|
| resnet18(Image-level) | 0.948 | 0.988  | 0.994 |    1    | 0.988 | 0.994 | 0.979  | 0.929 |  0.979  |    1     |   0.978   | 0.816 | 0.850 |   0.875    |   0.896    | 0.956  |
| resnet18(Pixel-level) | 0.962 | 0.990  | 0.986 |  0.995  | 0.971 | 0.955 |   1    | 0.937 |  0.954  |  0.986   |   0.961   | 0.905 | 0.985 |   0.990    |   0.830    | 0.984  |

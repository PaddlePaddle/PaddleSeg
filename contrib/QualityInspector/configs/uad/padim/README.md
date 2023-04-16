# PaDiM-Anomaly-Detection-Localization
此模型是使用Paddle复现论文: [PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization](https://arxiv.org/pdf/2011.08785).

![image](https://user-images.githubusercontent.com/61047836/227188224-62885b21-80fe-476c-aa81-c1181ced9626.png)

PaDiM是一种基于表示的无监督异常检测算法，在训练时，输入图像通过预训练的CNN骨干网络得到不同尺度的特征图，特征图经过随机选择维度的方式进行降维后拟合多元高斯分布；在推理时，使用马氏距离衡量每个特征图位置的异常分数，分数越高表示异常区域的可能性越高。

## MVTec AD数据集上的实验结果

* 图像级和像素级ROCAUC指标:


|                       |   Avg   | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
|-----------------------|:-------:| :----: | :---: |:-------:|:-----:|:-----:|:------:|:-----:|:-------:|:--------:|:---------:|:-----:| :---: |:----------:| :--------: | :----: |
| resnet18(Image-level) |  0.918  | 0.996  | 0.928 |   1     | 0.965 | 0.981 | 0.999  | 0.876 |  0.875  |  0.760   |   0.984   | 0.844 | 0.774 |   0.969    |   0.972    | 0.850  |
| resnet18(Pixel-level) |  0.962  | 0.990  | 0.944 |  0.987  | 0.894 | 0.931 | 0.979  | 0.949 |  0.980  |  0.970   |   0.963   | 0.923 | 0.975 |   0.985    |   0.974    | 0.981  |

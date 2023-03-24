# PatchCore
此模型是使用Paddle复现论文: [Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/pdf/2106.08265v2.pdf).

![image](https://user-images.githubusercontent.com/61047836/227188059-6ea90766-496d-432f-936f-f371c1f2fc0d.png)

PatchCore是一种基于表示的无监督异常检测算法，在训练时，输入图像通过预训练的CNN骨干网络得到不同尺度的特征图，特征图经过KNN Greedy CoreSet 采样选取最具代表性的特征点，构建特征向量记忆池；在推理时，通过比对测试图像的特征与特征向量记忆池的距离得到每个特征位置的异常分数。

## MVTec AD数据集上的实验结果

* 图像级和像素级ROCAUC指标:

|                       |   Avg   | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
|-----------------------|:-------:| :----: | :---: |:-------:|:-----:|:-----:|:------:|:-----:|:-------:|:--------:|:---------:|:-----:| :---: |:----------:| :--------: | :----: |
| resnet18(Image-level) |  0.979  | 0.992  | 0.971 |    1    | 0.990 | 0.990 |   1    | 0.979 |  0.982  |  0.999   |   0.997   | 0.939 | 0.947 |   0.942    |   0.993    | 0.971  |
| resnet18(Pixel-level) |  0.978  | 0.991  | 0.978 |  0.999  | 0.937 | 0.942 | 0.982  | 0.985 |  0.989  |  0.988   |   0.984   | 0.976 | 0.994 |   0.991    |   0.957    | 0.987  |
| resnet18(PRO_score)   |  0.932  | 0.961  | 0.923 |  0.971  | 0.848 | 0.893 | 0.947  | 0.943 |  0.937  |  0.935   |   0.939   | 0.928 | 0.971 |   0.918    |   0.910    | 0.957  |

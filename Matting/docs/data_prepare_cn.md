# 数据准备

当需要对自定义的数据集进行训练的时候，需要按照相应的格式进行准备。我们提供了两种数据形式，一种为离线合成，一种为在线合成。

## 离线合成
如果图像已经实现离线合成或者不需要合成，需按照如下模型整理数据结构。
```
dataset_root/
|--train/
|  |--fg/
|  |--alpha/
|
|--val/
|  |--fg/
|  |--alpha/
|
|--train.txt
|
|--val.txt
```
其中，fg目录下存放原图，另外fg目录下的图象名称需和alpha目录下的名称一一对应, 且两者的分辨率需保持一致。

train.txt和val.txt的内容如下
```
train/fg/14299313536_ea3e61076c_o.jpg
train/fg/14429083354_23c8fddff5_o.jpg
train/fg/14559969490_d33552a324_o.jpg
...
```

## 在线合成
数据读取支持在线合成，即输入网络的原图通过已有的前景图、alpha和背景图进行在线合成。类似[Deep Image Matting](https://arxiv.org/pdf/1703.03872.pdf)论文里使用的数据集Composition-1k，则数据集应整理成如下结构：
```
Composition-1k/
|--bg/
|
|--train/
|  |--fg/
|  |--alpha/
|
|--val/
|  |--fg/
|  |--alpha/
|  |--trimap/ (如果存在)
|
|--train.txt
|
|--val.txt
```

其中，fg目录存放前景图片，bg存放背景图片。

train.txt的内容如下：
```
train/fg/fg1.jpg bg/bg1.jpg
train/fg/fg2.jpg bg/bg2.jpg
train/fg/fg3.jpg bg/bg3.jpg
...
```
其中第一列为前景图像，第二列为背景图。

val.txt的内容如下, 如果不存在对应的trimap，则第三列可不提供，代码将会自动生成。
```
val/fg/fg1.jpg bg/bg1.jpg val/trimap/trimap1.jpg
val/fg/fg2.jpg bg/bg2.jpg val/trimap/trimap2.jpg
val/fg/fg3.jpg bg/bg3.jpg val/trimap/trimap3.jpg
...
```

# PaddleSeg 分割模型介绍

- [U-Net](#U-Net)
- [DeepLabv3+](#DeepLabv3\+)
- [ICNet](#ICNet)
- [PSPNet](#PSPNet)

## U-Net
U-Net [1] 起源于医疗图像分割，整个网络是标准的encoder-decoder网络，特点是参数少，计算快，应用性强，对于一般场景适应度很高。U-Net最早于2015年提出，并在ISBI 2015 Cell Tracking Challenge取得了第一。经过发展，目前有多个变形和应用。

原始U-Net的结构如下图所示，由于网络整体结构类似于大写的英文字母U，故得名U-net。左侧可视为一个编码器，右侧可视为一个解码器。编码器有四个子模块，每个子模块包含两个卷积层，每个子模块之后通过max pool进行下采样。由于卷积使用的是valid模式，故实际输出比输入图像小一些。具体来说，后一个子模块的分辨率=(前一个子模块的分辨率-4)/2。U-Net使用了Overlap-tile 策略用于补全输入图像的上下信息，使得任意大小的输入图像都可获得无缝分割。同样解码器也包含四个子模块，分辨率通过上采样操作依次上升，直到与输入图像的分辨率基本一致。该网络还使用了跳跃连接，以拼接的方式将解码器和编码器中相同分辨率的feature map进行特征融合，帮助解码器更好地恢复目标的细节。

![](./imgs/unet.png)

## DeepLabv3+

DeepLabv3+ [2] 是DeepLab系列的最后一篇文章，其前作有 DeepLabv1，DeepLabv2, DeepLabv3,
在最新作中，DeepLab的作者通过encoder-decoder进行多尺度信息的融合，同时保留了原来的空洞卷积和ASSP层，
其骨干网络使用了Xception模型，提高了语义分割的健壮性和运行速率，在 PASCAL VOC 2012 dataset取得新的state-of-art performance，89.0mIOU。

![](./imgs/deeplabv3p.png)

在PaddleSeg当前实现中，支持两种分类Backbone网络的切换

- MobileNetv2:
适用于移动设备的快速网络，如果对分割性能有较高的要求，请使用这一backbone网络。

- Xception:
DeepLabv3+原始实现的backbone网络，兼顾了精度和性能，适用于服务端部署。


## ICNet

Image Cascade Network（ICNet) [3] 主要用于图像实时语义分割。相较于其它压缩计算的方法，ICNet即考虑了速度，也考虑了准确性。 ICNet的主要思想是将输入图像变换为不同的分辨率，然后用不同计算复杂度的子网络计算不同分辨率的输入，然后将结果合并。ICNet由三个子网络组成，计算复杂度高的网络处理低分辨率输入，计算复杂度低的网络处理分辨率高的网络，通过这种方式在高分辨率图像的准确性和低复杂度网络的效率之间获得平衡。

整个网络结构如下：

![](./imgs/icnet.png)

## PSPNet

Pyramid Scene Parsing Network (PSPNet) [4] 起源于场景解析(Scene Parsing)领域。如下图所示，普通FCN [5] 面向复杂场景出现三种误分割现象：（1）关系不匹配。将船误分类成车，显然车一般不会出现在水面上。（2）类别混淆。摩天大厦和建筑物这两个类别相近，误将摩天大厦分类成建筑物。（3）类别不显著。枕头区域较小且纹理与床相近，误将枕头分类成床。

![](./imgs/pspnet2.png)

PSPNet的出发点是在算法中引入更多的上下文信息来解决上述问题。为了融合了图像中不同区域的上下文信息，PSPNet通过特殊设计的全局均值池化操作（global average pooling）和特征融合构造金字塔池化模块 (Pyramid Pooling Module)。PSPNet最终获得了2016年ImageNet场景解析挑战赛的冠军，并在PASCAL VOC 2012 和 Cityscapes 数据集上取得当时的最佳效果。整个网络结构如下：

![](./imgs/pspnet.png)

## 参考文献

[1] [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

[2] [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)

[3] [ICNet for Real-Time Semantic Segmentation on High-Resolution Images](https://arxiv.org/abs/1704.08545)

[4] [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)

[5] [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)

[回到开始](PaddleSeg 分割模型介绍)

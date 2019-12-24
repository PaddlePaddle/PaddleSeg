# PaddleSeg 分割模型介绍

### U-Net
U-Net [1] 起源于医疗图像分割，整个网络是标准的encoder-decoder网络，特点是参数少，计算快，应用性强，对于一般场景适应度很高。
![](./imgs/unet.png)

### DeepLabv3+

DeepLabv3+ [2] 是DeepLab系列的最后一篇文章，其前作有 DeepLabv1，DeepLabv2, DeepLabv3,
在最新作中，DeepLab的作者通过encoder-decoder进行多尺度信息的融合，同时保留了原来的空洞卷积和ASSP层，
其骨干网络使用了Xception模型，提高了语义分割的健壮性和运行速率，在 PASCAL VOC 2012 dataset取得新的state-of-art performance，89.0mIOU。

![](./imgs/deeplabv3p.png)

在PaddleSeg当前实现中，支持两种分类Backbone网络的切换

- MobileNetv2:
适用于移动设备的快速网络，如果对分割性能有较高的要求，请使用这一backbone网络。

- Xception:
DeepLabv3+原始实现的backbone网络，兼顾了精度和性能，适用于服务端部署。


### ICNet

Image Cascade Network（ICNet) [3] 主要用于图像实时语义分割。相较于其它压缩计算的方法，ICNet即考虑了速度，也考虑了准确性。 ICNet的主要思想是将输入图像变换为不同的分辨率，然后用不同计算复杂度的子网络计算不同分辨率的输入，然后将结果合并。ICNet由三个子网络组成，计算复杂度高的网络处理低分辨率输入，计算复杂度低的网络处理分辨率高的网络，通过这种方式在高分辨率图像的准确性和低复杂度网络的效率之间获得平衡。

整个网络结构如下：

![](./imgs/icnet.png)

### PSPNet

Pyramid Scene Parsing Network (PSPNet) [4] 

## 参考文献

1. [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)

2. [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

3. [ICNet for Real-Time Semantic Segmentation on High-Resolution Images](https://arxiv.org/abs/1704.08545)






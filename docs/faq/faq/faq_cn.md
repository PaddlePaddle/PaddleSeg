简体中文 | [English](faq.md)
# FAQ

## Q1: PaddleSeg 如何从本地加载预训练模型的权重参数？

* **Answer**:PaddleSeg 模型的推荐配置参数统一存放在 PaddleSeg/configs 下各个模型文件夹的 yaml 文件中。比如 ANN 的其中一个配置在 /PaddleSeg/configs/ann/ann_resnet50_os8_cityscapes_1024x512_80k.yml 中给出。如下图所示：

![](./faq_imgs/ann_config.png)

> 图中红色部分为骨干网络的预训练模型参数文件的存放位置。**请注意**：此处将直接以 https 链接形式下载我们提供的预训练模型参数。如果你在本地拥有骨干网络的预训练模型参数，请用其存放的绝对路径替换该 yaml 文件中 `backbone` 下的 `pretrained`。或者，你可以根据将要执行的 `train.py` 所在的目录为该预训练模型参数的存放位置设置相对路径。

> 图中绿色部分为分割网络的预训练模型参数文件的存放位置。如果你在本地拥有分割网络的预训练模型参数，请用其存放的绝对路径替换该 yaml 文件中的 `pretrained`。

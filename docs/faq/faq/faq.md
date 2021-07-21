English | [简体中文](faq_cn.md)
# FAQ

## Q1: How to load the weight parameters of the pre-trained model locally？

* **Answer**:The recommended configuration parameters of the model are stored in the yaml file of each model folder under PaddleSeg/configs. For example, one of the configurations of ANN is given in /PaddleSeg/configs/ann/ann_resnet50_os8_cityscapes_1024x512_80k.yml. As shown below:

![](./faq_imgs/ann_config.png)

> The red part in the figure is the storage location of the pre-training model parameter file of the backbone network. **Note**: Here, we will download the pre-training model parameters provided by us directly in the form of a https link. If you have the pre-trained model parameters of the backbone network locally, please replace the `pretrained` under `backbone` in the yaml file with the absolute path. Or, you should set the relative path for the storage location of the pre-training model parameters according to the directory where the `train.py` will be executed.

> The green part in the figure is the storage location of the pre-training model parameter file of the segmentation network. If you have the pre-trained model parameters of the segmentation network locally, please replace the `pretrained` in the yaml file with the absolute path or relative path where it is stored.
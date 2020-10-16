# 数据集准备

PaddleSeg目前支持CityScapes、ADE20K、Pascal VOC等数据集的加载，在加载数据集时，如若本地不存在对应数据，则会自动触发下载

## 关于CityScapes数据集

由于协议限制，请自行前往[CityScapes官网](https://www.cityscapes-dataset.com/)下载数据集，我们建议您将数据集存放于`PaddleSeg/dygraph/data`中，以便与我们配置文件完全兼容

## 关于Pascal VOC数据集



## 自定义数据集

如果您需要使用自定义数据集进行训练，请按照以下步骤准备数据：

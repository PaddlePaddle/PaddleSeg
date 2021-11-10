# 医疗相关

以下内容为EISeg中医疗垂类相关的文档，主要包括环境配置和功能介绍两大方面。

## 1 环境配置

EISeg使用SimpleITK打开医学影像，安装方式如下：

```shell
pip install SimpleITK
```

## 2 功能介绍

目前EISeg只支持打开单层的Dicom格式图像，对Nitfi格式和多张Dicom格式的支持正在开发中。如果打开图像的后缀为 .dcm 会询问是否开启医疗组件。

![med-prompt](https://linhandev.github.io/assets/img/post/Med/med-prompt.png)

点击确定后会出现图像窗宽窗位的设置

![med-widget](https://linhandev.github.io/assets/img/post/Med/med-widget.png)

目前EISeg提供[肝脏分割预训练模型](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18s_ocr48_lits.zip)，推荐窗宽窗位400, 0。该模型用于肝脏分割效果最佳，但也可以用于其他组织或器官的分割。

简体中文 | [English](medical_en.md)

# 医疗相关

以下内容为EISeg中医疗垂类相关的文档，主要包括环境配置和功能介绍。

## 1 环境配置

使用医疗组件需要额外安装SimpleITK包用于读取医学影像，安装方式如下：

```shell
pip install SimpleITK
```

## 2 功能介绍

目前EISeg支持打开**单层的Dicom格式图像**，对Nitfi格式和多张Dicom的支持正在开发中。EISeg通过图像拓展名判断图像格式。打开单张图像时需要在右下角类型下拉菜单中选择医疗图像，如下图所示

打开文件夹时和自然图像过程相同。打开 .dcm 后缀的图像后会询问是否开启医疗组件。

![med-prompt](https://linhandev.github.io/assets/img/post/Med/med-prompt.png)

点击确定后会出现图像窗宽窗位设置面板

![med-widget](https://linhandev.github.io/assets/img/post/Med/med-widget.png)

窗宽窗位的作用是聚焦一定的强度区间，方便观察CT扫描。CT扫描中每个像素点存储的数值代表人体在该位置的密度，密度越高数值越大，图像的数据范围通常为-1024～1024。不过查看扫描时人眼无法分辨2048个灰度，因此通常选择一个更小的强度范围，将这一区间内图像的灰度差异拉大，从而方便观察。具体的操作是取扫描中强度范围在 窗位-窗宽/2～窗位+窗宽/2 的部分，将这一部分数据放入256灰度的图片中展示给用户。

推理方面，目前EISeg针对医疗场景提供[肝脏分割预训练模型](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18s_ocr48_lits.zip)，推荐窗宽窗位400, 0。该模型用于肝脏分割效果最佳，也可以用于其他组织或器官的分割。

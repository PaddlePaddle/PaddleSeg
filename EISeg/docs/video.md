简体中文 | [English](video_en.md)

# 视频及3D医疗标注相关

以下内容为EISeg中交互式视频标注垂类相关的文档，主要包括模型选择，数据准备和使用步骤。

## 环境配置

使用3D显示需要额外安装VTK用于3D医学影像显示，安装方式如下：

```shell
pip install vtk
```

## 使用演示

![dance](https://user-images.githubusercontent.com/35907364/175504795-d41f0842-cb18-4675-9763-3e817f168edf.gif)

## 模型选择

EISeg视频标注工具以EISeg交互式分割算法及[MIVOS](https://github.com/hkchengrex/MiVOS)交互式视频分割算法为基础，基于Paddle开发的一个高效的图像及视频标注软件。
它涵盖了通用、腹腔多器官，CT椎骨等不同方向的高质量交互式视频分割模型，方便开发者快速实现视频的标注，降低标注成本。 对于3D医疗标注的尝试，我们将医疗切片数据视作视频帧关系，利用帧间传播实现3D医疗图像的标注。结合EISeg已有的高精度交互式分割算法，进一步扩展了视频分割算法的使用边界。

在使用EISeg前，请先下载传播模型参数。用户需要根据自己的场景需求选择对应的交互式分割模型及传播模型。若您想使用3D显示功能，可在`显示`菜单中勾选3D显示功能。

![lits](https://user-images.githubusercontent.com/35907364/178422205-40327d43-c7d4-4a5d-87fb-63c08308fb9f.gif)


| 模型类型  | 适用场景                   | 模型结构       | 模型下载地址                                                     | 配套传播模型下载地址 |
| -------- | -------------------------- | -------------- | ------------------------------------------------------------ |-------------|
| 高精度模型 | 通用场景的图像标注 | HRNet18_OCR64  | [static_hrnet18_ocr64_cocolvis](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18_ocr64_cocolvis.zip) | [static_propagation](https://www.wjx.cn/vm/OrTuFZA.aspx)         |
| 轻量化模型 | 通用场景的图像标注 | HRNet18s_OCR48 | [static_hrnet18s_ocr48_cocolvis](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18s_ocr48_cocolvis.zip) | [static_propagation](https://www.wjx.cn/vm/OrTuFZA.aspx)        |
| 高精度模型 | 通用图像标注场景      | EdgeFlow | [static_edgeflow_cocolvis](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_edgeflow_cocolvis.zip) | [static_propagation](https://www.wjx.cn/vm/OrTuFZA.aspx)        |
| 高精度模型 | 人像标注场景      | HRNet18_OCR64  | [static_hrnet18_ocr64_human](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18_ocr64_human.zip) | [static_propagation](https://www.wjx.cn/vm/OrTuFZA.aspx)        |
| 轻量化模型 | 人像标注场景       | HRNet18s_OCR48 | [static_hrnet18s_ocr48_human](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18s_ocr48_human.zip) | [static_propagation](https://www.wjx.cn/vm/OrTuFZA.aspx)       |
| 轻量化模型 | 医疗肝脏标注场景       | HRNet18s_OCR48 | [static_hrnet18s_ocr48_lits](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18s_ocr48_lits.zip) | [static_propagation_lits](https://www.wjx.cn/vm/OrTuFZA.aspx)         |
| 轻量化模型 | CT椎骨图像标注场景       | HRNet18s_OCR48 | [static_hrnet18s_ocr48_MRSpineSeg](https://paddleseg.bj.bcebos.com/eiseg/0.5/static_hrnet18s_ocr48_MRSpineSeg.zip) | [static_propagation_spine](https://www.wjx.cn/vm/OrTuFZA.aspx)        |

## 数据准备
- 由于视频处理计算量较大，推荐使用带显卡带机器进行视频分割和3D医疗图片带标注，并且标注图像帧数不宜超过100帧，若视频超过该帧数，可以通过[cut_video.py](../tool/cut_video.py)来进行视频截取。
- 对于3D医疗图像的标注是基于视频传播进行的，因此在标注前请先将切片图像转换成视频格式，脚本为[medical2video.py](../tool/medical2video.py)。

## 使用步骤

1. **模型参数加载**

   根据标注场景，选择合适的网络模型及参数进行加载。选择合适的模型及参数下载解压后，模型结构`*.pdmodel`及相应的模型参数`*.pdiparams`需要放到同一个目录下，加载模型时只需选择`*.pdiparams`结尾的模型参数位置即可。静态图模型初始化时间稍长，请耐心等待模型加载完成后进行下一步操作。正确加载的模型参数会记录在`近期模型参数`中，可以方便切换，并且下次打开软件时自动加载退出时的模型参数。

2. **图像加载**

   打开图像/图像文件夹。当看到主界面图像正确加载，`数据列表`正确出现图像路径即可。加载视频格式的文件后，EIVSeg会自动弹出视频标注相关组件。**由于视频标注计算量较大，请确保在具有显卡的机器上进行视频标注,推荐每个视频帧长在100帧以内，如不符合，可用[cut_video.py](../tool/cut_video.py)进行视频分段**。

3. **标签添加/加载**

   添加/加载标签。可以通过`添加标签`新建标签，标签分为4列，分别对应像素值、说明、颜色和删除。新建好的标签可以通过`保存标签列表`保存为txt文件，其他合作者可以通过`加载标签列表`将标签导入。通过加载方式导入的标签，重启软件后会自动加载。
在视频标注时，需要将标签完全确定，**标签设定尽可能覆盖所有待标注类别，否则会影响视频传播的结果**。

4. **传播模型参数加载**

    根据标注场景，选择上方提供的与EISeg模型匹配的传播模型。下载完模型解压后，随意选择其中一个以`*.pdiparams`结尾的模型参数位置即可，不需要对解压后的模型及参数名称进行修改，否则会导致加载出错。

5. **交互式分割确定参考帧**

    利用鼠标左键(添加)或右键(删除)所选择的区域，最终点击空格键获取参考帧分割结果，注意**标注图像时尽可能覆盖所有待标注及传播类别，否则会影响视频传播的结果**。

6. **帧传播**

   按下视频组件中的`传播`按钮，模型会自动计算与参考帧相似区域并生成标注结果。

7. **修改**

   如果对中间结果不满意，重复5-6步骤。

8. **保存**

   点击左下角保存按钮，选择保存路径即可将结果保存。

# PaddleSeg 产业实践

提供基于PaddlSeg最新的分割特色模型:

- [人像分割](./HumanSeg)
- [遥感分割](./RemoteSensing)
- [人体解析](./ACE2P)
- [车道线分割](./LaneNet)
- [工业表盘分割](#工业表盘分割)
- [AIStudio在线教程](#AIStudio在线教程)

## 人像分割 HumanSeg

HumanSeg系列全新升级，提供三个适用于不同场景，包含适用于移动端实时分割场景的模型`HumanSeg-lite`，提供了包含光流的后处理的优化，使人像分割在视频场景中更加顺畅，更多详情请参考[HumanSeg](./HumanSeg)

## 遥感分割 Remote Sensing Segmentation
PaddleSeg遥感影像分割涵盖图像预处理、数据增强、模型训练、预测流程。
针对遥感数据多通道、分布范围大、分布不均的特点，我们支持多通道训练预测，内置10+多通道预处理和数据增强的策略，可结合实际业务场景进行定制组合，提升模型泛化能力和鲁棒性。
内置U-Net, HRNet两种主流分割网络，可选择不同的损失函数如Dice Loss, BCE Loss等方式强化小目标和不均衡样本场景下的分割精度。更多详情请参考[RemoteSensing](./RemoteSensing)

以下是遥感云检测的示例效果：

![](./RemoteSensing/docs/imgs/vis.png)

## 人体解析 Human Parsing

人体解析(Human Parsing)是细粒度的语义分割任务，旨在识别像素级别的人类图像的组成部分（例如，身体部位和服装）。ACE2P通过融合底层特征、全局上下文信息和边缘细节，端到端训练学习人体解析任务。以ACE2P单人人体解析网络为基础的解决方案在CVPR2019第三届LIP挑战赛中赢得了全部三个人体解析任务的第一名


#### ACE2P模型框架图
![](./ACE2P/imgs/net.jpg)

PaddleSeg提供了ACE2P获得比赛冠军的预训练模型，更多详情请点击[ACE2P](./ACE2P)

## 车道线分割 LaneNet

PaddleSeg提供了基于LaneNet的车道线分割模型，更多详情请点击[LaneNet](./LaneNet)

![](https://pic2.zhimg.com/80/v2-8015f4b256791d4456fbc2739efc106d_1440w.jpg)


## 工业表盘分割


**Note:** 本章节所有命令均在`PaddleSeg`目录下执行。

### 1. 模型结构

U-Net

### 2. 数据准备

执行以下命令下载并解压数据集，数据集将存放在contrib/MechanicalIndustryMeter文件夹下：

```
python ./contrib/MechanicalIndustryMeter/download_mini_mechanical_industry_meter.py
```

### 3. 下载预训练模型

```
python ./pretrained_model/download_model.py unet_bn_coco
```

### 4. 训练与评估

```
export CUDA_VISIBLE_DEVICES=0
python ./pdseg/train.py --log_steps 10 --cfg contrib/MechanicalIndustryMeter/unet_mechanical_meter.yaml --use_gpu --do_eval --use_mpio
```

### 5. 可视化
我们已提供了一个训练好的模型，执行以下命令进行下载，下载后将存放在./contrib/MechanicalIndustryMeter/文件夹下。

```
python ./contrib/MechanicalIndustryMeter/download_unet_mechanical_industry_meter.py
```

使用该模型进行预测可视化：

```
python ./pdseg/vis.py --cfg contrib/MechanicalIndustryMeter/unet_mechanical_meter.yaml --use_gpu --vis_dir vis_meter \
TEST.TEST_MODEL "./contrib/MechanicalIndustryMeter/unet_mechanical_industry_meter/"
```
可视化结果会保存在./vis_meter文件夹下。

### 6. 可视化结果示例：

  原图：

  ![](MechanicalIndustryMeter/imgs/1560143028.5_IMG_3091.JPG)

  预测结果：

  ![](MechanicalIndustryMeter/imgs/1560143028.5_IMG_3091.png)

## AIStudio在线教程

PaddleSeg在AI Studio平台上提供了在线体验的教程，欢迎体验：

|教程|链接|
|-|-|
|工业质检|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/184392)|
|人像分割|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/475345)|
|特色垂类模型|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/226710)|

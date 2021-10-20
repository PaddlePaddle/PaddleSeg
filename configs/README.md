# 模型库

### CNN系列

|模型\骨干网络|ResNet50|ResNet101|HRNetw18|HRNetw48|
|-|-|-|-|-|
|[ANN](./ann)|✔|✔|||
|[BiSeNetv2](./bisenet)|-|-|-|-|
|[DANet](./danet)|✔|✔|||
|[Deeplabv3](./deeplabv3)|✔|✔|||
|[Deeplabv3P](./deeplabv3p)|✔|✔|||
|[Fast-SCNN](./fastscnn)|-|-|-|-|
|[FCN](./fcn)|||✔|✔|
|[GCNet](./gcnet)|✔|✔|||
|[GSCNN](./gscnn)|✔|✔|||
|[HarDNet](./hardnet)|-|-|-|-|
|[OCRNet](./ocrnet/)|||✔|✔|
|[PSPNet](./pspnet)|✔|✔|||
|[U-Net](./unet)|-|-|-|-|
|[U<sup>2</sup>-Net](./u2net)|-|-|-|-|
|[Att U-Net](./attention_unet)|-|-|-|-|
|[U-Net++](./unet_plusplus)|-|-|-|-|
|[U-Net3+](./unet_3plus)|-|-|-|-|
|[DecoupledSegNet](./decoupled_segnet)|✔|✔|||
|[EMANet](./emanet)|✔|✔|-|-|
|[ISANet](./isanet)|✔|✔|-|-|
|[DNLNet](./dnlnet)|✔|✔|-|-|
|[SFNet](./sfnet)|✔|-|-|-|
|[PP-HumanSeg-Lite](./pp_humanseg_lite)|-|-|-|-|
|[PortraitNet](./portraitnet)|-|-|-|-|
|[STDC](./stdcseg)|-|-|-|-|
|[GINet](./ginet)|✔|✔|-|-|
|[PointRend](./pointrend)|✔|✔|-|-|
|[SegNet](./segnet)|-|-|-|-|

### Transformer系列
* [SETR](./setr)
* [MLATransformer](../contrib/AutoNUE/configs)
* [SegFormer](./segformer)

# 模型说明

<div align="center">
<img src="../docs/images/model_performance.jpg"   width = "700"//>  
</div>

# 模型性能参数


|Model|Backbone|Resolution|Training Iters|mIoU|mIoU(flip)|mIoU(ms+flip)|inference_time(ms)|pre_process_time(ms)|post_process_time(ms)
|-|-|-|-|-|-|-|-|-|-|
|ANN|ResNet101|1024x512|80000|79.50%|79.77%|79.69%|94.91|143.35|0.013
|BiSeNetv2|-|1024x1024|160000|73.19%|74.19%|74.43%|16.00|167.45|0.013
|DANet|ResNet50|1024x512|80000|80.27%|80.53%|-|95.08|134.78|0.015
|Deeplabv3|ResNet101_OS8|1024x512|80000|80.85%|81.09%|81.54%|114|141.65|0.014
|Deeplabv3P|ResNet50_OS8|1024x512|80000|81.10%|81.38%|81.24%|69.78|147.24|0.016
|Fast-SCNN|-|1024x1024|160000|69.31%|-|-|10.43|161.52|0.012
|FCN|HRNet_W48|1024x512|80000|80.70%|81.24%|81.56%|45.46|130.58|0.012
|GCNet|ResNet101_OS8|1024x512|80000|81.01%|81.30%|81.64%|90.28|119.38|0.013
|GSCNN|ResNet50_OS8|1024x512|80000|80.67%|80.88%|80.88%|-|-|-
|HarDNet|-|1024x1024|160000|79.03%|79.49%|79.76%|21.19|164.36|0.013
|OCRNet|HRNet_W48|1024x512|160000|82.15%|82.59%|82.85%|61.88|138.48|0.014
|PSPNet|ResNet101_OS8|1024x512|80000|80.48%|80.74%|81.04%|115.93|115.94|0.012
|U-Net|-|1024x512|160000|65.00%|66.02%|66.89%|29.11|137.75|0.012
|U^2-Net|-|1024x512|160000|-|-|-|-|-|-
|Att U-Net|-|1024x512|160000|-|-|-|-|-|-|-
|U-Net++|-|1024x512|160000|-|-|-|-|-|-
|DecoupledSegNet|ResNet50_OS8|1024x512|80000|81.26%|81.56%|81.80%|66.89|136.28|0.013
|EMANet|ResNet101_OS8|1024x512|80000|80.00%|80.23%|80.53%|80.05|140.47|0.013
|ISANet|ResNet101_OS8|769x769|80000|80.10%|80.30%|80.26%|91.72|129.12|0.012
|DNLNet|ResNet101_OS8|1024x512|80000|81.03%|81.38%|-|97.81|138.95|0.014
|SFNet|ResNet50_OS8|1024x1024|80000|81.49%|81.63%|81.85%|121.35|160.45|0.013


- 以上性能参数均在Tesla V100 16GB下得到。
- 该表展示了PaddleSeg所实现的分割模型在取得最高分类精度的配置下的一些评价参数。
- 推理时间是使用CityScapes数据集中的图像进行100次预测取平均值的结果。
- 其中，mIoU、mIoU(flip)、mIoU(ms+flip)是对模型进行评估的结果。`ms` 表示**multi-scale**，即使用三种scale [0.75, 1.0, 1.25]；`flip`表示水平翻转。



# 配置项

----
### train_dataset
>  训练数据集
>
>  * 参数
>     * type : 数据集类型，所支持值请参考训练配置文件
>     * **others** : 请参考对应模型训练配置文件

----
### val_dataset
>  评估数据集
>  * 参数
>     * type : 数据集类型，所支持值请参考训练配置文件
>     * **others** : 请参考对应模型训练配置文件
>

----
### batch_size
>  单张卡上，每步迭代训练时的数据量

----
### iters
>  训练步数

----
### optimizer
> 训练优化器
>  * 参数
>     * type : 优化器类型，支持目前Paddle官方所有优化器
>     * weight_decay : L2正则化的值
>     * **others** : 请参考[Paddle官方Optimizer文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Overview_cn.html)

----
### lr_scheduler
> 学习率
>  * 参数
>     * type : 学习率类型，支持10种策略，分别是'PolynomialDecay', 'PiecewiseDecay', 'StepDecay', 'CosineAnnealingDecay', 'ExponentialDecay', 'InverseTimeDecay', 'LinearWarmup', 'MultiStepDecay', 'NaturalExpDecay', 'NoamDecay'.
>     * **others** : 请参考[Paddle官方LRScheduler文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/LRScheduler_cn.html)

----
### learning_rate（不推荐使用该配置，将来会被废弃，建议使用`lr_scheduler`代替）
> 学习率
>  * 参数
>     * value : 初始学习率
>     * decay : 衰减配置
>       * type : 衰减类型，目前只支持poly
>       * power : 衰减率
>       * end_lr : 最终学习率

----
### loss
> 损失函数
>  * 参数
>     * types : 损失函数列表
>       * type : 损失函数类型，所支持值请参考损失函数库
>       * ignore_index : 训练过程需要忽略的类别，默认取值与`train_dataset`的ignore_index一致，**推荐不用设置此项**。如果设置了此项，`loss`和`train_dataset`的ignore_index必须相同。
>     * coef : 对应损失函数列表的系数列表

----
### model
> 待训练模型
>  * 参数
>     * type : 模型类型，所支持值请参考模型库
>     * **others** : 请参考对应模型训练配置文件
---
### export
> 模型导出配置
>  * 参数
>    * transforms : 预测时的预处理操作，支持配置的transforms与`train_dataset`、`val_dataset`等相同。如果不填写该项，默认只会对数据进行归一化标准化操作。

具体配置文件说明请参照[配置文件详解](../docs/design/use/use_cn.md)

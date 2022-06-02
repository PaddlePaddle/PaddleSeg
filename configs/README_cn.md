大量模型的配置文件保存在`PaddleSeg/configs`目录下。 PaddleSeg使用这些配置文件进行模型训练、测试和导出。

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

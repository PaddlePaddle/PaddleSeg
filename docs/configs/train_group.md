# cfg.TRAIN

TRAIN Group存放所有和训练相关的配置

## `MODEL_SAVE_DIR`
在训练周期内定期保存模型的主目录

### 默认值
无（需要用户自己填写）

<br/>
<br/>

## `PRETRAINED_MODEL_DIR`
预训练模型路径

### 默认值
无

### 注意事项

* 若未指定该字段，则模型会随机初始化所有的参数，从头开始训练

* 若指定了该字段，但是路径不存在，则参数加载失败，仍然会被随机初始化

* 若指定了该字段，且路径存在，但是部分参数不存在或者shape无法对应，则该部分参数随机初始化

<br/>
<br/>

## `RESUME_MODEL_DIR`
从指定路径中恢复参数并继续训练

### 默认值
无

### 注意事项

* 当`RESUME_MODEL_DIR`存在时，PaddleSeg会恢复到上一次训练的最近一个epoch，并且恢复训练过程中的临时变量（如已经衰减过的学习率，Optimizer的动量数据等），`PRETRAINED_MODEL`路径的最后一个目录必须为int数值或者字符串final，PaddleSeg会将int数值作为当前起始EPOCH继续训练，若目录为final，则不会继续训练。若目录不满足上述条件，PaddleSeg会抛出错误。

<br/>
<br/>

## `SYNC_BATCH_NORM`
是否在多卡间同步BN的均值和方差。

Synchronized Batch Norm跨GPU批归一化策略最早在[MegDet: A Large Mini-Batch Object Detector](https://arxiv.org/abs/1711.07240)
论文中提出，在[Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/pdf/1902.04103.pdf)论文中以Yolov3验证了这一策略的有效性。

PaddleSeg基于PaddlePaddle框架的sync_batch_norm策略，可以支持通过多卡实现大batch size的分割模型训练，可以得到更高的mIoU精度。

### 默认值
False

### 注意事项

* 打开该选项会带来一定的性能消耗（多卡间同步数据导致）

* 仅在GPU多卡训练时该开关有效（Windows不支持多卡训练，因此无需打开该开关）

* GPU多卡训练时，建议开启该开关，可以提升模型的训练效果

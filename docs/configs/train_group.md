# cfg.TRAIN

TRAIN Group存放所有和训练相关的配置

## `MODEL_SAVE_DIR`
在训练周期内定期保存模型的主目录

## 默认值
无（需要用户自己填写）

<br/>
<br/>

## `PRETRAINED_MODEL`
预训练模型路径

## 默认值
无

## 注意事项

* 若未指定该字段，则模型会随机初始化所有的参数，从头开始训练

* 若指定了该字段，但是路径不存在，则参数加载失败，仍然会被随机初始化

* 若指定了该字段，且路径存在，但是部分参数不存在或者shape无法对应，则该部分参数随机初始化

<br/>
<br/>

## `RESUME`
是否从预训练模型中恢复参数并继续训练

## 默认值
False

## 注意事项

* 当该字段被置为True且`PRETRAINED_MODEL`不存在时，该选项不生效

* 当该字段被置为True且`PRETRAINED_MODEL`存在时，PaddleSeg会恢复到上一次训练的最近一个epoch，并且恢复训练过程中的临时变量（如已经衰减过的学习率，Optimizer的动量数据等）

* 当该字段被置为True且`PRETRAINED_MODEL`存在时，`PRETRAINED_MODEL`路径的最后一个目录必须为int数值或者字符串final，PaddleSeg会将int数值作为当前起始EPOCH继续训练，若目录为final，则不会继续训练。若目录不满足上述条件，PaddleSeg会抛出错误。

<br/>
<br/>

## `SYNC_BATCH_NORM`
是否在多卡间同步BN的均值和方差

## 默认值
False

## 注意事项

* 打开该选项会带来一定的性能消耗（多卡间同步数据导致）

* 仅在GPU多卡训练时该开关有效（Windows不支持多卡训练，因此无需打开该开关）

* GPU多卡训练时，建议开启该开关，可以提升模型的训练效果
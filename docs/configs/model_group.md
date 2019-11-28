# cfg.MODEL

MODEL Group存放所有和模型相关的配置，该Group还包含三个子Group

* [DeepLabv3p](./model_deeplabv3p_group.md)
* [UNet](./model_unet_group.md)
* [ICNet](./model_icnet_group.md)
* [HRNet](./model_hrnet_group.md)

## `MODEL_NAME`

所选模型，支持`deeplabv3p` `unet` `icnet` `hrnet`四种模型

### 默认值

无（需要用户自己填写）

<br/>
<br/>

## `DEFAULT_NORM_TYPE`

模型所用norm类型，支持`bn` [`gn`]()

### 默认值

`bn`

<br/>
<br/>

## `DEFAULT_GROUP_NUMBER`

默认GROUP数量，仅在`DEFAULT_NORM_TYPE`为`gn`时生效

### 默认值

32

<br/>
<br/>

## `BN_MOMENTUM`

BatchNorm动量, 一般无需改动

### 默认值

0.99

<br/>
<br/>

## `DEFAULT_EPSILON`

BatchNorm计算时所用的极小值, 防止分母除0溢出，一般无需改动

### 默认值

1e-5

<br/>
<br/>

## `FP16`

是否开启FP16训练

### 默认值

False

<br/>
<br/>

## `SCALE_LOSS`

对损失进行缩放的系数

### 默认值

1.0

### 注意事项
* 启动fp16训练时，建议设置该字段为8

<br/>
<br/>

## `MULTI_LOSS_WEIGHT`

多路损失的权重

### 默认值

[1.0]

### 注意事项

* 该字段仅在模型存在多路损失的情况下生效

* 目前支持的模型中只有`icnet`使用多路（3路）损失

* 当选择模型为`icnet`且该字段的长度不为3时，PaddleSeg会强制设置该字段为[1.0, 0.4, 0.16]

### 示例
假设模型存在三路损失，计算结果分别为loss1/loss2/loss3，并且`MULTI_LOSS_WEIGHT`的值为[1.0, 0.4, 0.16]，则最终损失的计算结果为
```math
loss = 1.0 * loss1 + 0.4 * loss2 + 0.16 * loss3
```

<br/>
<br/>


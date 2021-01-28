# cfg.MODEL.DEEPLAB

MODEL.DEEPLAB 子Group存放所有和DeepLabv3+模型相关的配置

## `BACKBONE`

DeepLabV3+所用骨干网络，支持`mobilenetv2` `xception65` `xception41` `resnet50_vd` `resnet101_vd`

### 默认值

`xception65`

<br/>
<br/>

## `OUTPUT_STRIDE`

DeepLabV3+下采样率，支持8/16两种选择

### 默认值

16

<br/>
<br/>

## `DEPTH_MULTIPER`

MobileNet V2的depth mutiper值，仅当`BACKBONE`为`mobilenetv2`生效

### 默认值

1.0

<br/>
<br/>

## `ENCODER_WITH_ASPP`

DeepLabv3+的模型Encoder中是否使用ASPP

### 默认值

True

### 注意事项
* 将该功能置为False可以提升模型计算速度，但是会降低精度

<br/>
<br/>

## `DECODER_WITH_ASPP`

DeepLabv3+的模型是否使用Decoder

### 默认值

True

### 注意事项
* 将该功能置为False可以提升模型计算速度，但是会降低精度

<br/>
<br/>

## `ASPP_WITH_SEP_CONV`

DeepLabv3+的模型的ASPP模块是否使用可分离卷积

### 默认值

False

<br/>
<br/>

## `DECODER_WITH_SEP_CONV`

DeepLabv3+的模型的Decoder模块是否使用可分离卷积

### 默认值

False

<br/>
<br/>

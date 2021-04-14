# cfg.MODEL.UNET

MODEL.UNET 子Group存放所有和UNet模型相关的配置

## `UPSAMPLE_MODE`

上采样方式，支持`bilinear`或者不设置

### 默认值

`bilinear`

### 注意事项
* 当`UPSAMPLE_MODE`值为`bilinear`时，UNet上采样方法为双线性插值法，否则使用转置卷积进行上采样

<br/>
<br/>

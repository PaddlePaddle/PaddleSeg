# 实时人像分割预测部署

本模型基于飞浆开源的人像分割模型，并做了大量的针对视频的光流追踪优化，提供了完整的支持视频流的实时人像分割解决方案，并提供了高性能的`Python`和`C++`集成部署方案，以满足不同场景的需求。


## 模型下载

支持的模型文件如下，请根据应用场景选择合适的模型：

|模型文件 | 说明 |
| --- | --- |
|[shv75_deeplab_0303_quant](https://paddleseg.bj.bcebos.com/deploy/models/shv75_0303_quant.zip)  | 小模型, 适合轻量级计算环境 |
|[shv75_deeplab_0303](https://paddleseg.bj.bcebos.com/deploy/models/shv75_deeplab_0303.zip)| 小模型，适合轻量级计算环境 |
|[deeplabv3_xception_humanseg](https://paddleseg.bj.bcebos.com/deploy/models/deeplabv3_xception_humanseg.zip) | 服务端GPU环境 |

**注意：下载后解压到合适的路径，后续该路径将做为预测参数用于加载模型。**


## 预测部署
- [Python预测部署](./python)

## 效果预览

<figure class="half">
    <img src="https://paddleseg.bj.bcebos.com/deploy/data/input.gif">
    <img src="https://paddleseg.bj.bcebos.com/deploy/data/output.gif">
</figure>

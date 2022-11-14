# 模型库
针对高频应用场景——人像抠图，开源了高质量人像抠图模型库，可根据实际应用场景直接部署应用，也可进行微调训练。

- PP-Matting高精度抠图模型，通过引导流设计实现语义引导下高分辨率图像抠图。追求更高精度，推荐使用该模型。
    且该模行提供了512和1024两个分辨率级别的预训练模型。
- PP-MattingV2轻量级抠图模型，通过双层金字塔池化及空间注意力提取高级语义信息，并利用多级特征融合机制兼顾语义和细节的预测。
    对比MODNet模型推理速度提升44.6%， 误差平均相对减小17.91。追求更高速度，推荐使用该模型。

| 模型 | SAD | MSE | Grad | Conn |Params(M) | FLOPs(G) | FPS | Config File | Checkpoint | Inference Model |
| - | - | -| - | - | - | - | -| - | - |
| PP-MattingV2-512   |40.59|0.0038|33.86|38.90| 8.95 | 7.51 | 98.89 |[cfg](../configs/ppmattingv2/ppmattingv2-stdc1-human_512.yml)| [model](https://paddleseg.bj.bcebos.com/matting/models/ppmattingv2-stdc1-human_512.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/ppmattingv2-stdc1-human_512.zip) |
| PP-Matting-512     |31.56|0.0022|31.80|30.13| 24.5 | 91.28 | 28.9 |[cfg](../configs/ppmatting/ppmatting-hrnet_w18-human_512.yml)| [model](https://paddleseg.bj.bcebos.com/matting/models/ppmatting-hrnet_w18-human_512.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/ppmatting-hrnet_w18-human_512.zip) |
| PP-Matting-1024    |66.22|0.0088|32.90|64.80| 24.5 | 91.28 | 13.4(1024X1024) |[cfg](../configs/ppmatting/ppmatting-hrnet_w18-human_1024.yml)| [model](https://paddleseg.bj.bcebos.com/matting/models/ppmatting-hrnet_w18-human_1024.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/ppmatting-hrnet_w18-human_1024.zip) |
| PP-HumanMatting    |53.15|0.0054|43.75|52.03| 63.9 | 135.8 (2048X2048)| 32.8(2048X2048)|[cfg](../configs/human_matting/human_matting-resnet34_vd.yml)| [model](https://paddleseg.bj.bcebos.com/matting/models/human_matting-resnet34_vd.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/pp-humanmatting-resnet34_vd.zip) |
| MODNet-MobileNetV2 |50.07|0.0053|35.55|48.37| 6.5 | 15.7 | 68.4 |[cfg](../configs/modnet/modnet-mobilenetv2.yml)| [model](https://paddleseg.bj.bcebos.com/matting/models/modnet-mobilenetv2.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/modnet-mobilenetv2.zip) |
| MODNet-ResNet50_vd |39.01|0.0038|32.29|37.38| 92.2 | 151.6 | 29.0 |[cfg](../configs/modnet/modnet-resnet50_vd.yml)| [model](https://paddleseg.bj.bcebos.com/matting/models/modnet-resnet50_vd.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/modnet-resnet50_vd.zip) |
| MODNet-HRNet_W18   |35.55|0.0035|31.73|34.07| 10.2 | 28.5 | 62.6 |[cfg](../configs/modnet/modnet-hrnet_w18.yml)| [model](https://paddleseg.bj.bcebos.com/matting/models/modnet-hrnet_w18.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/modnet-hrnet_w18.zip) |
| DIM-VGG16          |32.31|0.0233|28.89|31.45| 28.4 | 175.5| 30.4 |[cfg](../configs/dim/dim-vgg16.yml)| [model](https://paddleseg.bj.bcebos.com/matting/models/dim-vgg16.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/dim-vgg16.zip) |

**注意**：
* 指标计算数据集为PPM-100和AIM-500中的人像部分共同组成，共195张，[PPM-AIM-195](https://paddleseg.bj.bcebos.com/matting/datasets/PPM-AIM-195.zip)。
* FLOPs和FPS计算默认模型输入大小为(512, 512), GPU为Tesla V100 32G。FPS基于Paddle Inference预测裤进行计算。
* DIM为trimap-based的抠图方法，指标只计算过度区域部分，对于没有提供Trimap的情况下，默认将0<alpha<255的区域以25像素为半径进行膨胀腐蚀后作为过度区域。

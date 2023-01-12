English | [简体中文](model_zoo_overview_cn.md)

# Model Zoo

## Semantic Segmentation Model Zoo

PaddleSeg provides 45+ semantic segmentation models, 150+ well-trained models, 10+ backbones.

In [`PaddleSeg/configs`](../configs), we provide the config files and readme.md for all models on common dataset, e.g., [PP-LiteSeg](../configs/pp_liteseg/).
Besides, the readme.md file introduces the origin paper, the performance and the trained weights.

Some common models are as follows.

**CNN Series**

|Model\Backbone Network|ResNet50|ResNet101|HRNetw18|HRNetw48|
|-|-|-|-|-|
|[ANN](../configs/ann)|✔|✔|||
|[BiSeNetv2](../configs/bisenet)|-|-|-|-|
|[DANet](../configs/danet)|✔|✔|||
|[Deeplabv3](../configs/deeplabv3)|✔|✔|||
|[Deeplabv3P](../configs/deeplabv3p)|✔|✔|||
|[Fast-SCNN](../configs/fastscnn)|-|-|-|-|
|[FCN](../configs/fcn)|||✔|✔|
|[GCNet](../configs/gcnet)|✔|✔|||
|[GSCNN](../configs/gscnn)|✔|✔|||
|[HarDNet](../configs/hardnet)|-|-|-|-|
|[OCRNet](../configs/ocrnet/)|||✔|✔|
|[PSPNet](../configs/pspnet)|✔|✔|||
|[U-Net](../configs/unet)|-|-|-|-|
|[U<sup>2</sup>-Net](../configs/u2net)|-|-|-|-|
|[Att U-Net](../configs/attention_unet)|-|-|-|-|
|[U-Net++](../configs/unet_plusplus)|-|-|-|-|
|[U-Net3+](../configs/unet_3plus)|-|-|-|-|
|[DecoupledSegNet](../configs/decoupled_segnet)|✔|✔|||
|[EMANet](../configs/emanet)|✔|✔|-|-|
|[ISANet](../configs/isanet)|✔|✔|-|-|
|[DNLNet](../configs/dnlnet)|✔|✔|-|-|
|[SFNet](../configs/sfnet)|✔|-|-|-|
|[PP-HumanSeg-Lite](../configs/pp_humanseg_lite)|-|-|-|-|
|[PortraitNet](../configs/portraitnet)|-|-|-|-|
|[STDC](../configs/stdcseg)|-|-|-|-|
|[GINet](../configs/ginet)|✔|✔|-|-|
|[PointRend](../configs/pointrend)|✔|✔|-|-|
|[SegNet](../configs/segnet)|-|-|-|-|
|[ESPNetV2](../configs/espnet)|-|-|-|-|
|[HRNetW48Contrast](../configs/hrnet_w48_contrast)|-|-|-|✔|
|[DMNet](../configs/dmnet)|-|✔|-|-|
|[ESPNetV1](../configs/espnetv1)|-|-|-|-|
|[ENCNet](../configs/encnet)|-|✔|-|-|
|[PFPNNet](../configs/pfpn)|-|✔|-|-|
|[FastFCN](../configs/fastfcn)|✔|-|-|-|
|[BiSeNetV1](../configs/bisenetv1)|-|-|-|-|
|[ENet](../configs/enet)|-|-|-|-|
|[CCNet](../configs/ccnet)|-|✔|-|-|
|[DDRNet](../configs/ddrnet)|-|-|-|-|
|[GloRe](../configs/glore)|✔|-|-|-|
|[PP-LiteSeg](../configs/pp_liteseg)|-|-|-|-|

**Transformer series**
* [SETR](../configs/setr)
* [MLATransformer](../contrib/AutoNUE/configs)
* [SegFormer](../configs/segformer)
* [SegMenter](../configs/segmenter)

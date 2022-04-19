简体中文 | [English](model_zoo_overview.md)

# PaddleSeg模型库总览

## 模型库
### CNN系列

|模型\骨干网络|ResNet50|ResNet101|HRNetw18|HRNetw48|
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

### Transformer系列
* [SETR](../configs/setr)
* [MLATransformer](../contrib/AutoNUE/configs)
* [SegFormer](../configs/segformer)
* [SegMenter](../configs/segmenter)

# 模型库Benchmark
基于Cityscapes数据集，PaddleSeg支持22+系列分割算法以及对应的30+个图像分割预训练模型，性能评估如下。

**测试环境：**

- GPU: Tesla V100 16GB
- CPU: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
- CUDA: 10.2
- cuDNN: 7.6
- Paddle: 2.1.3
- PaddleSeg: 2.3

**测试方法:**

- 单GPU，Batch size为1，运行耗时为纯模型预测时间，预测图片尺寸为1024x512。
- 模型导出后使用Paddle Inference的Python API测试。
- 推理时间是使用CityScapes数据集中的100张图像进行预测取平均值的结果。
- 部分算法只测试了取得最高分割精度的配置下的模型性能。

## 精度 vs 速度
<div align="center">
<img src=https://user-images.githubusercontent.com/30695251/140323144-c44671ac-8ff8-4c11-a6f7-1ea339c27852.png //>  
</div>

## 精度 vs FLOPs
<div align="center">
<img src=https://user-images.githubusercontent.com/30695251/140323107-02ce9de4-c8f4-4f18-88b2-59bd0055a70b.png //>  
</div>

## 精度 vs Params
<div align="center">
<img src=https://user-images.githubusercontent.com/30695251/140323131-ed03fbb1-a583-47f5-a7dd-f4ea3582c345.png //>  
</div>

## 总表
|Model|Backbone|mIoU|Flops(G)|Params(M)|Inference Time(ms)|Preprocess Time(ms)|Postprocess Time(ms)
|-|-|-|-|-|-|-|-|
|BiSeNetv2|-|73.19%|16.14|2.33|16.00|167.45|0.013
|Fast-SCNN|-|69.31%|2.04|1.44|10.43|161.52|0.012
|HarDNet|-|79.03%|35.40|4.13|21.19|164.36|0.013
|U-Net|-|65.00%|253.75|13.41|29.11|137.75|0.012
|SegFormer_B0|-|76.73%|13.63|3.72|15.66|152.60|0.017
|SegFormer_B1|-|78.35%|26.55|13.68|21.48|152.40|0.017
|STDC1-Seg50|STDC1|74.74%|24.83|8.29|9.10|153.01|0.016
|STDC2-Seg50|STDC2|77.60%|38.05|12.33|10.88|152.64|0.015
|ANN|ResNet101|79.50%|564.43|67.70|94.91|143.35|0.013
|DANet|ResNet50|80.27%|398.48|47.52|95.08|134.78|0.015
|Deeplabv3|ResNet101_OS8|80.85%|481.00|58.17|114|141.65|0.014
|Deeplabv3P|ResNet50_OS8|81.10%|228.44|26.79|69.78|147.24|0.016
|FCN|HRNet_W48|80.70%|187.50|65.94|45.46|130.58|0.012
|GCNet|ResNet101_OS8|81.01%|570.74|68.73|90.28|119.38|0.013
|OCRNet|HRNet_W48|82.15%|324.66|70.47|61.88|138.48|0.014
|PSPNet|ResNet101_OS8|80.48%|686.89|86.97|115.93|115.94|0.012
|DecoupledSegNet|ResNet50_OS8|81.26%|395.10|41.71|66.89|136.28|0.013
|EMANet|ResNet101_OS8|80.00%|512.18|61.45|80.05|140.47|0.013
|ISANet|ResNet101_OS8|80.10%|474.13|56.81|91.72|129.12|0.012
|DNLNet|ResNet101_OS8|81.03%|575.04|69.13|97.81|138.95|0.014
|SFNet|ResNet18_OS8|78.72%|136.80|13.81|69.51|131.67|0.015
|SFNet|ResNet50_OS8|81.49%|394.37|42.03|121.35|160.45|0.013
|PointRend|ResNet50_OS8|76.54%|363.17|28.18|70.35|157.24|0.016
|SegFormer_B2|-|81.60%|113.71|27.36|47.08|155.45|0.016
|SegFormer_B3|-|82.47%|142.97|47.24|62.70|154.68|0.017
|SegFormer_B4|-|82.38%|171.05|64.01|73.26|151.11|0.017
|SegFormer_B5|-|82.58%|199.68|84.61|84.34|147.92|0.016
|SETR-Naive|Vision Transformer|77.29%|620.94|303.37|201.26|145.76|0.016
|SETR-PUP|Vision Transformer|78.08%|727.46|307.24|212.22|147.05|0.016
|SETR-MLA|Vision Transformer|76.52%|633.88|307.05|204.87|145.87|0.015


<!-- |GINet|ResNet50_OS8|78.66%|463.36|55.87|-|-|-
|GINet|ResNet101_OS8|78.4%|618.95|74.91|-|-|-
|GSCNN|ResNet50_OS8|80.67%|385.50|39.47|-|-|- -->

## 如何添加新模型到Benchmark
### 性能数据统计
按照上述配置搭建测试环境，并按测试方法要求进行测试。其中Inference Time(ms)，Preprocess Time(ms)，Postprocess Time(ms)耗时可通过 [PaddleSeg推理部署教程](deployment/inference/python_inference_cn.md)进行测试，开启`--benchmark`参数进行推理即可。

### 图表绘制
将所获得的性能数据更新至表格中。性能对比图绘制代码位于`PaddleSeg/tools/plot_model_performance.py`，在set_model_info()中补充模型的性能数据，运行
```python
python plot_model_performance.py
```
即可得到所有的性能对比图

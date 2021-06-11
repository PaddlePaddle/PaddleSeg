# PaddleSeg Benchmark with AMP

## 动态图
数据集cityscapes 放置于data目录下, 下载链接：https://paddleseg.bj.bcebos.com/dataset/cityscapes.tar

通过 **--fp16** 开启amp训练。

<img src="./docs/images/seg_news_icon.png" width="50"/> *[2021-02-26] PaddleSeg has released the v2.0 version, which supports the dynamic graph by default. The static-graph codes have been moved to [legacy](./legacy). See detailed [release notes](./docs/release_notes.md).*

![demo](./docs/images/cityscapes.gif)

Welcome to PaddleSeg! PaddleSeg is an end-to-end image segmentation development kit developed based on [PaddlePaddle](https://www.paddlepaddle.org.cn), which covers a large number of high-quality segmentation models in different directions such as *high-performance* and *lightweight*. With the help of modular design, we provide two application methods: *Configuration Drive* and *API Calling*. So one can conveniently complete the entire image segmentation application from training to deployment through configuration calls or API calls.

## Core features

**High performance model**: Based on the high-performance backbone trained by Baidu's self-developed [semi-supervised label knowledge distillation scheme (SSLD)](https://paddleclas.readthedocs.io/zh_CN/latest/advanced_tutorials/distillation/distillation.html#ssld), combined with the state of the art segmentation technology, we provides 50+ high-quality pre-training models, which are better than other open source implementations.

**Modular design**: PaddleSeg support 15+ mainstream *segmentation networks*, developers can start based on actual application scenarios and assemble diversified training configurations combined with modular design of *data enhancement strategies*, *backbone networks*, *loss functions* and other different components to meet different performance and accuracy requirements.

**High efficiency**: PaddleSeg provides multi-process asynchronous I/O, multi-card parallel training, evaluation and other acceleration strategies, combined with the memory optimization function of the PaddlePaddle, which can greatly reduce the training overhead of the segmentation model, all this allowing developers to lower cost and more efficiently train image segmentation model.

## Model Zoo

|Model\Backbone|ResNet50|ResNet101|HRNetw18|HRNetw48|
|-|-|-|-|-|
|[ANN](./configs/ann)|✔|✔|||
|[BiSeNetv2](./configs/bisenet)|-|-|-|-|
|[DANet](./configs/danet)|✔|✔|||
|[Deeplabv3](./configs/deeplabv3)|✔|✔|||
|[Deeplabv3P](./configs/deeplabv3p)|✔|✔|||
|[Fast-SCNN](./configs/fastscnn)|-|-|-|-|
|[FCN](./configs/fcn)|||✔|✔|
|[GCNet](./configs/gcnet)|✔|✔|||
|[GSCNN](./configs/gscnn)|✔|✔|||
|[HarDNet](./configs/hardnet)|-|-|-|-|
|[OCRNet](./configs/ocrnet/)|||✔|✔|
|[PSPNet](./configs/pspnet)|✔|✔|||
|[U-Net](./configs/unet)|-|-|-|-|
|[U<sup>2</sup>-Net](./configs/u2net)|-|-|-|-|
|[Att U-Net](./configs/attention_unet)|-|-|-|-|
|[U-Net++](./configs/unet_plusplus)|-|-|-|-|
|[U-Net3+](./configs/unet_3plus)|-|-|-|-|
|[DecoupledSegNet](./configs/decoupled_segnet)|✔|✔|||
|[EMANet](./configs/emanet)|✔|✔|-|-|
|[ISANet](./configs/isanet)|✔|✔|-|-|
|[DNLNet](./configs/dnlnet)|✔|✔|-|-|
|[SFNet](./configs/sfnet)|✔|-|-|-|
|[ShuffleNetV2](./configs/shufflenetv2)|-|-|-|-|

## Dataset

DeepLabv3+ 模型的配置文件为：
benchmark/deeplabv3p.yml

**注意**

* 动态图中batch_size设置为每卡的batch_size
* DeepLabv3+ 支持通过传入 **--data_format NHWC**进行‘NHWC’数据格式的训练。



## 静态图
数据集cityscapes 放置于legacy/dataset目录下

通过 **MODEL.FP16 True** 开启amp训练
单机单卡使用如下命令进行训练：
```
cd legacy
export CUDA_VISIBLE_DEVICES=0
python pdseg/train.py --cfg configs/hrnetw18_cityscapes_1024x512_215.yaml --use_gpu  --use_mpio --log_steps 10 BATCH_SIZE 2 SOLVER.NUM_EPOCHS 3 MODEL.FP16 True
```

单机多卡使用如下命令进行训练：
```
export CUDA_VISIBLE_DEVICES=0,1
fleetrun pdseg/train.py --cfg configs/hrnetw18_cityscapes_1024x512_215.yaml --use_gpu  --use_mpio --log_steps 10 BATCH_SIZE 4 SOLVER.NUM_EPOCHS 3 MODEL.FP16 True
```

deeplabv3p模型的配置文件为：
configs/deeplabv3p_resnet50_vd_cityscapes.yaml

**注意**
静态图中的BATCH_SIZE为总的batch size。

## 竞品
竞品为[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

对应竞品配置文件为：

configs/hrnet/fcn_hr18_512x1024_80k_cityscapes.py

## Tutorials

* [Get Started](./docs/quick_start.md)
* [API Tutorial](https://aistudio.baidu.com/aistudio/projectdetail/1339458)
* [Data Preparation](./docs/data_prepare.md)
* [Training Configuration](./configs/)
* [Loss Usage](./docs/loss_usage.md)
* [API References](./docs/apis)
* [Add New Components](./docs/add_new_model.md)
* [Model Compression](./slim)
* [Model Deploy](./docs/model_export.md)

## Practical Cases

* [HumanSeg](./contrib/HumanSeg)
* [Cityscapes SOTA](./contrib/CityscapesSOTA)

## Feedbacks and Contact
* The dynamic version is still under development, if you find any issue or have an idea on new features, please don't hesitate to contact us via [GitHub Issues](https://github.com/PaddlePaddle/PaddleSeg/issues).
* PaddleSeg User Group (QQ): 1004738029 or 850378321 or 793114768

## Acknowledgement
* Thanks [jm12138](https://github.com/jm12138) for contributing U<sup>2</sup>-Net.
* Thanks [zjhellofss](https://github.com/zjhellofss) (Fu Shenshen) for contributing Attention U-Net, and Dice Loss.
* Thanks [liuguoyu666](https://github.com/liguoyu666), [geoyee](https://github.com/geoyee) for contributing U-Net++ and U-Net3+.
* Thanks [yazheng0307](https://github.com/yazheng0307) (LIU Zheng) for contributing quick-start document.

## Citation
If you find our project useful in your research, please consider citing:

```latex
@misc{liu2021paddleseg,
      title={PaddleSeg: A High-Efficient Development Toolkit for Image Segmentation},
      author={Yi Liu and Lutao Chu and Guowei Chen and Zewu Wu and Zeyu Chen and Baohua Lai and Yuying Hao},
      year={2021},
      eprint={2101.06175},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{paddleseg2019,
    title={PaddleSeg, End-to-end image segmentation kit based on PaddlePaddle},
    author={PaddlePaddle Authors},
    howpublished = {\url{https://github.com/PaddlePaddle/PaddleSeg}},
    year={2019}
}
```

# Lovasz loss
对于图像分割任务中，经常出现类别分布不均匀的情况，例如：工业产品的瑕疵检测、道路提取及病变区域提取等。我们可使用lovasz loss解决这个问题。

Lovasz loss基于子模损失(submodular losses)的凸Lovasz扩展，对神经网络的mean IoU损失进行优化。Lovasz loss根据分割目标的类别数量可分为两种：lovasz hinge loss和lovasz softmax loss. 其中lovasz hinge loss适用于二分类问题，lovasz softmax loss适用于多分类问题。该工作发表在CVPR 2018上，可点击[参考文献](#参考文献)查看具体原理。


## Lovasz hinge loss
### 使用指南

PaddleSeg通过`cfg.SOLVER.LOSS`参数可以选择训练时的损失函数，
如`cfg.SOLVER.LOSS=['lovasz_hinge_loss','bce_loss']`将指定训练loss为`lovasz hinge loss`与`bce loss`(binary cross-entropy loss)的组合。

Lovasz hinge loss有3种使用方式：（1）直接训练使用。（2）bce loss结合使用。（3）先使用bec loss进行训练，再使用lovasz hinge loss进行finetuning. 第1种方式不一定达到理想效果，推荐使用后两种方式。本文以第2种方式为例。

同时，也可以通过`cfg.SOLVER.LOSS_WEIGHT`参数对不同loss进行权重配比，灵活运用于训练调参。如下所示
```yaml
SOLVER:
    LOSS: ["lovasz_hinge_loss","bce_loss"]
    LOSS_WEIGHT:
        LOVASZ_HINGE_LOSS: 0.5
        BCE_LOSS: 0.5
```

### 实验对比

我们以道路提取任务为例应用lovasz hinge loss.
基于MiniDeepGlobeRoadExtraction数据集与bce loss进行了实验对比。
该数据集来源于DeepGlobe比赛的Road Extraction单项，训练数据道路占比为：4.5%. 如下为其图片样例：
<p align="center">
  <img src="./imgs/deepglobe.png" hspace='10'/> <br />
 </p>
可以看出道路在整张图片中的比例很小。

为进行快速体验，这里使用DeepLabv3+模型，backbone为MobileNetV2.

* 数据集下载
我们从DeepGlobe比赛的Road Extraction的训练集中随机抽取了800张图片作为训练集，200张图片作为验证集，
制作了一个小型的道路提取数据集[MiniDeepGlobeRoadExtraction](https://paddleseg.bj.bcebos.com/dataset/MiniDeepGlobeRoadExtraction.zip)

```shell
python dataset/download_mini_deepglobe_road_extraction.py
```

* 预训练模型下载
```shell
python pretrained_model/download_model.py deeplabv3p_mobilenetv2-1-0_bn_coco
```
* 配置/数据校验
```shell
python pdseg/check.py --cfg ./configs/lovasz_hinge_deeplabv3p_mobilenet_road.yaml
```

* 训练
```shell
python pdseg/train.py --cfg ./configs/lovasz_hinge_deeplabv3p_mobilenet_road.yaml --use_gpu --use_mpio SOLVER.LOSS "['lovasz_hinge_loss','bce_loss']"
```

* 评估
```shell
python pdseg/eval.py --cfg ./configs/lovasz_hinge_deeplabv3p_mobilenet_road.yaml --use_gpu --use_mpio SOLVER.LOSS "['lovasz_hinge_loss','bce_loss']"
```

* 结果比较

lovasz hinge loss + bce loss和softmax loss的对比结果如下图所示。
<p align="center">
  <img src="./imgs/lovasz-hinge.png" hspace='10'/> <br />
 </p>

图中蓝色曲线为lovasz hinge loss + bce loss，最高mIoU为76.2%，橙色曲线为softmax loss， 最高mIoU为73.44%，相比提升2.76个百分点。



## Lovasz softmax loss
### 使用指南

PaddleSeg通过`cfg.SOLVER.LOSS`参数可以选择训练时的损失函数，
如`cfg.SOLVER.LOSS=['lovasz_softmax_loss','softmax_loss']`将指定训练loss为`lovasz softmax loss`与`softmax loss`的组合。

Lovasz softmax loss有3种使用方式：（1）直接训练使用。（2）softmax loss结合使用。（3）先使用softmax loss进行训练，再使用lovasz softmax loss进行finetuning. 第1种方式不一定达到理想效果，推荐使用后两种方式。本文以第2种方式为例。

同时，也可以通过`cfg.SOLVER.LOSS_WEIGHT`参数对不同loss进行权重配比，灵活运用于训练调参。如下所示
```yaml
SOLVER:
    LOSS: ["lovasz_softmax_loss","softmax_loss"]
    LOSS_WEIGHT:
        LOVASZ_SOFTMAX_LOSS: 0.2
        SOFTMAX_LOSS: 0.8
```

### 实验对比

接下来以PASCAL VOC 2012数据集为例应用lovasz softmax loss. 我们将lovasz softmax loss与softmax loss进行了实验对比。为进行快速体验，这里使用DeepLabv3+模型，backbone为MobileNetV2.


* 数据集下载
<p align="center">
  <img src="./imgs/VOC2012.png" width="50%" height="50%" hspace='10'/> <br />
 </p>

```shell
python dataset/download_and_convert_voc2012.py
```

* 预训练模型下载
```shell
python pretrained_model/download_model.py deeplabv3p_mobilenetv2-1-0_bn_coco
```
* 配置/数据校验
```shell
python pdseg/check.py --cfg ./configs/lovasz_softmax_deeplabv3p_mobilenet_pascal.yaml
```

* 训练
```shell
python pdseg/train.py --cfg ./configs/lovasz_softmax_deeplabv3p_mobilenet_pascal.yaml --use_gpu --use_mpio SOLVER.LOSS "['lovasz_softmax_loss','softmax_loss']"

```

* 评估
```shell
python pdseg/eval.py --cfg ./configs/lovasz_softmax_deeplabv3p_mobilenet_pascal.yaml --use_gpu --use_mpio SOLVER.LOSS "['lovasz_softmax_loss','softmax_loss']"

```

* 结果比较

lovasz softmax loss + softmax loss和softmax loss的对比结果如下图所示。
<p align="center">
  <img src="./imgs/lovasz-softmax.png" hspace='10' /> <br />
 </p>

图中橙色曲线代表lovasz softmax loss + softmax loss，最高mIoU为64.63%，蓝色曲线代表softmax loss， 最高mIoU为63.55%，相比提升1.08个百分点。


## 参考文献
[Berman M, Rannen Triki A, Blaschko M B. The lovász-softmax loss: a tractable surrogate for the optimization of the intersection-over-union measure in neural networks[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 4413-4421.](http://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.html)

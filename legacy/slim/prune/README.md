# PaddleSeg剪裁教程

在阅读本教程前，请确保您已经了解过[PaddleSeg使用说明](../../docs/usage.md)等章节，以便对PaddleSeg有一定的了解

该文档介绍如何使用[PaddleSlim](https://paddlepaddle.github.io/PaddleSlim)的卷积通道剪裁接口对检测库中的模型的卷积层的通道数进行剪裁。

在分割库中，可以直接调用`PaddleSeg/slim/prune/train_prune.py`脚本实现剪裁，在该脚本中调用了PaddleSlim的[paddleslim.prune.Pruner](https://paddlepaddle.github.io/PaddleSlim/api/prune_api/#Pruner)接口。

该教程中所示操作，如无特殊说明，均在`PaddleSeg/`路径下执行。

## 1. 数据与预训练模型准备
执行如下命令，下载cityscapes数据集
```
python dataset/download_cityscapes.py
```
参照[预训练模型列表](../../docs/model_zoo.md)获取所需预训练模型

## 2. 确定待分析参数

我们通过剪裁卷积层参数达到缩减卷积层通道数的目的，在剪裁之前，我们需要确定待裁卷积层的参数的名称。
通过以下命令查看当前模型的所有参数：

```python
# 查看模型所有Paramters
for x in train_prog.list_vars():
    if isinstance(x, fluid.framework.Parameter):
        print(x.name, x.shape)

```

通过观察参数名称和参数的形状，筛选出所有卷积层参数，并确定要裁剪的卷积层参数。

## 3. 启动剪裁任务

使用`train_prune.py`启动裁剪任务时，通过`SLIM.PRUNE_PARAMS`选项指定待裁剪的参数名称列表，参数名之间用逗号分隔，通过`SLIM.PRUNE_RATIOS`选项指定各个参数被裁掉的比例。

```shell
CUDA_VISIBLE_DEVICES=0
python -u ./slim/prune/train_prune.py --log_steps 10 --cfg configs/cityscape_fast_scnn.yaml --use_gpu --use_mpio \
SLIM.PRUNE_PARAMS 'learning_to_downsample/weights,learning_to_downsample/dsconv1/pointwise/weights,learning_to_downsample/dsconv2/pointwise/weights' \
SLIM.PRUNE_RATIOS '[0.1,0.1,0.1]'
```
这里我们选取三个参数，按0.1的比例剪裁。

## 4. 评估

```shell
CUDA_VISIBLE_DEVICES=0
python -u ./slim/prune/eval_prune.py --cfg configs/cityscape_fast_scnn.yaml --use_gpu \
TEST.TEST_MODEL your_trained_model \
```

## 5. 模型

| 模型 | 数据集合 | 下载地址 |剪裁方法| flops | mIoU on val|
|---|---|---|---|---|---|
| Fast-SCNN/bn | Cityscapes |[fast_scnn_cityscapes.tar](https://paddleseg.bj.bcebos.com/models/fast_scnn_cityscape.tar) | 无 | 7.21g | 0.6964 |
| Fast-SCNN/bn | Cityscapes |[fast_scnn_cityscapes-uniform-51.tar](https://paddleseg.bj.bcebos.com/models/fast_scnn_cityscape-uniform-51.tar) | uniform | 3.54g | 0.6990 |

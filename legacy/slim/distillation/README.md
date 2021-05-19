>运行该示例前请安装PaddleSlim和Paddle1.6或更高版本

# PaddleSeg蒸馏教程

在阅读本教程前，请确保您已经了解过[PaddleSeg使用说明](../../docs/usage.md)等章节，以便对PaddleSeg有一定的了解

该文档介绍如何使用[PaddleSlim](https://paddlepaddle.github.io/PaddleSlim)对分割库中的模型进行蒸馏。

该教程中所示操作，如无特殊说明，均在`PaddleSeg/`路径下执行。

## 概述

该示例使用PaddleSlim提供的[蒸馏策略](https://paddlepaddle.github.io/PaddleSlim/algo/algo/#3)对分割库中的模型进行蒸馏训练。
在阅读该示例前，建议您先了解以下内容：

- [PaddleSlim蒸馏API文档](https://paddlepaddle.github.io/PaddleSlim/api/single_distiller_api/)

## 安装PaddleSlim
可按照[PaddleSlim使用文档](https://paddlepaddle.github.io/PaddleSlim/)中的步骤安装PaddleSlim

## 蒸馏策略说明

关于蒸馏API如何使用您可以参考PaddleSlim蒸馏API文档

这里以Deeplabv3-xception蒸馏训练Deeplabv3-mobilenet模型为例，首先，为了对`student model`和`teacher model`有个总体的认识，进一步确认蒸馏的对象，我们通过以下命令分别观察两个网络变量（Variables）的名称和形状：

```python
# 观察student model的Variables
student_vars = []
for v in fluid.default_main_program().list_vars():
    try:
        student_vars.append((v.name, v.shape))
    except:
        pass
print("="*50+"student_model_vars"+"="*50)
print(student_vars)
# 观察teacher model的Variables
teacher_vars = []
for v in teacher_program.list_vars():
    try:
        teacher_vars.append((v.name, v.shape))
    except:
        pass
print("="*50+"teacher_model_vars"+"="*50)
print(teacher_vars)
```

经过对比可以发现，`student model`和`teacher model`输入到`loss`的特征图分别为：

```bash
# student model
bilinear_interp_0.tmp_0
# teacher model
bilinear_interp_2.tmp_0
```


它们形状两两相同，且分别处于两个网络的输出部分。所以，我们用`l2_loss`对这几个特征图两两对应添加蒸馏loss。需要注意的是，teacher的Variable在merge过程中被自动添加了一个`name_prefix`，所以这里也需要加上这个前缀`"teacher_"`，merge过程请参考[蒸馏API文档](https://paddlepaddle.github.io/PaddleSlim/api/single_distiller_api/#merge)

```python
distill_loss = l2_loss('teacher_bilinear_interp_2.tmp_0', 'bilinear_interp_0.tmp_0')
```

我们也可以根据上述操作为蒸馏策略选择其他loss，PaddleSlim支持的有`FSP_loss`, `L2_loss`, `softmax_with_cross_entropy_loss` 以及自定义的任何loss。

## 训练

根据[PaddleSeg/pdseg/train.py](../../pdseg/train.py)编写压缩脚本`train_distill.py`。
在该脚本中定义了teacher_model和student_model，用teacher_model的输出指导student_model的训练

### 执行示例

下载teacher的预训练模型([deeplabv3p_xception65_bn_cityscapes.tgz](https://paddleseg.bj.bcebos.com/models/xception65_bn_cityscapes.tgz))和student的预训练模型([mobilenet_cityscapes.tgz](https://paddleseg.bj.bcebos.com/models/mobilenet_cityscapes.tgz)),
修改student config file(./slim/distillation/cityscape.yaml)中预训练模型的路径:
```
TRAIN:
    PRETRAINED_MODEL_DIR: your_student_pretrained_model_dir
```
修改teacher config file(./slim/distillation/cityscape_teacher.yaml)中预训练模型的路径:
```
SLIM:
    KNOWLEDGE_DISTILL_TEACHER_MODEL_DIR: your_teacher_pretrained_model_dir
```

执行如下命令启动训练，每间隔```cfg.TRAIN.SNAPSHOT_EPOCH```会进行一次评估。
```shell
CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch ./slim/distillation/train_distill.py \
--log_steps 10 --cfg ./slim/distillation/cityscape.yaml \
--teacher_cfg ./slim/distillation/cityscape_teacher.yaml \
--use_gpu \
--do_eval
```

注意：如需修改配置文件中的参数，请在对应的配置文件中直接修改，暂不支持命令行输入覆盖。

## 评估预测

训练完成后的评估和预测请参考PaddleSeg的[快速入门](../../README.md#快速入门)和[基础功能](../../README.md#基础功能)等章节

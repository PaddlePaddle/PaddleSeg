简体中文 | [English](style.md)
# PaddleSeg模型开发规范

模型规范主要分为：新增文件的开发规范，可拓展模块的开发规范，导出和测试预测模型，新增模型的PR checklist。


## 1 新增文件的开发规范

每个新增文件都需要进行自查和修正，主要包含`copyright`部分，`import`部分和编码规范`checklist`。

### 1.1 `copyright` 部分

创建空文件```pspnet.py```后，在文件顶部添加以下`copyright`。

`PaddleSeg`中每个新增的文件都需要添加相应的版权信息，注意其中年份按照当前自然年改写。

```python
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

### 1.2 `import` 部分

该部分为导入模型所需要的package，所有Python文件需要按照以下顺序导入三个类型的`package`：
1. `Python`源生自带`package`；
2. 第三方`package`，即`pip`或`conda install`的`package`；
3. `PaddleSeg`中的`package`。

以下举例，注：不同类型`package`空行，删掉未使用的`package`，长度相差太大按照递增顺序排列。
```python
import os

import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils
```

### 1.3 编码自查 `checklist`

这部分主要面向 python 说明编码中需要注意的规范，其中大部分编码规范会有pre-commit进行校验修正，更多可以参考[谷歌编程规范](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/)。

- [ ] 空行：顶级定义之间空两行，比如函数或者类定义。 方法定义，类定义与第一个方法之间，都应该空一行。函数或方法中，某些地方要是你觉得是合适的逻辑中断，就空一行；

- [ ] 行长度：每行不超过80个字符，这代表分两屏之后可以完整看到所有代码。Python会将[圆括号, 中括号和花括号中的行隐式的连接起来](http://docs.python.org/2/reference/lexical_analysis.html#implicit-line-joining) ，你可以利用这个特点，在表达式外围增加一对额外的圆括号，而不要使用反斜杠换行；

- [ ] 括号：括号可以用于行连接，但是不要在各种判断中使用没有必要的括号；

- [ ] 分行：每个语句都要独立一行，不要使用分号。

- [ ] 命名：模块名写法: `module_name` ;包名写法: `package_name` ;类名: `ClassName` ;方法名: `method_name` ;异常名: `ExceptionName` ;函数名: `function_name` ;全局常量名: `GLOBAL_CONSTANT_NAME` ;全局变量名: `global_var_name` ;实例名: `instance_var_name` ;函数参数名: `function_parameter_name` ;局部变量名: `local_var_name`



## 2 可拓展模块的开发规范

目前`PaddleSeg`支持`model, loss, backbone, transform, dataset`的组件拓展，所有新增的组件都需要按照下面规范进行自查和修正。

backbone和模型的规范相近，transform的规范较为简单，因此下面主要对模型，损失和数据集进行规范说明。

### 2.1 模型实现部分

模型实现部分以PSPNet的开发为例进行说明。

开发`PSPNet`，需要在```paddleseg/models```目录下创建```pspnet.py```，文件名的字母均为小写。整个文件内容分为三个部分，`copyright`部分, `import`部分, 模型实现部分，前两部分参考前面`新增文件的开发规范`的说明。

模型实现的结构按顺序分为三个部分，主模型，分割头，辅助模块。若模型没有backbone则只有主模型和辅助模块，这里以三部分为例。

#### 2.1.1 主模型

**模型声明规范**

该部分在import部分之后，为模型实现的第一部分，即要求将主模型放在文件顶部。

1. 用 manager 完成主模型的添加，即在主模型类定义之前加上下列语句。注意，**只有**主模型需要manager修饰器，不可以通过函数来定义不同型号的主模型，推荐通过模型yml配置文件中的模型输入参数来定义不同型号的主模型。

```python
@manager.MODELS.add_component
class PSPNet(nn.Layer):
```

2. 主模型的class，需要继承 nn.Layer；

3. 添加英文注释
* 添加"`The xxx implementation based on PaddlePaddle.`";
* 添加"`The original article refers to` " + 作者名和文章名 + 论文链接；
* 添加输入参数的说明，需要指名参数类型，若可选则表明 `optional`，然后在该参数注释末尾添加"`Default: xx`"；
* 如果可以，可以进一步添加`Returns，Raises`说明函数/方法返回值和可能会有的报错。

```python
@manager.MODELS.add_component
class PSPNet(nn.Layer):
    """
    The PSPNet implementation based on PaddlePaddle.

    The original article refers to
    Zhao, Hengshuang, et al. "Pyramid scene parsing network"
    (https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): Backbone network, currently support Resnet50/101.
        backbone_indices (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
        pp_out_channels (int, optional): The output channels after Pyramid Pooling Module. Default: 1024.
        bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1,2,3,6).
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """
```

**__init__规范**

1. `__init__`中，参数列表的建议顺序为：`num_classes, backbone, backbone_indices, ......, align_corners, in_channels, pretrained`。参数的具体含义可以参考前面PSPNet示例的注释。前面参数若出现，则按照上面的顺序，其他中间参数顺序可以自由调整；
2. 参数名需要有含义，尽量避免无明显含义的名字，比如n, m, aa，除非原实现非用不可；
3. 参数全部显式写出，不能包括变长参数，比如:`*args, **kwargs`；
4. `super().__init__()`保持空参数；
5. 结尾调用```self.init_weight()```，加载`pretrained`指定的模型参数；
6. `in_channels`用来指定输入图片的通道数。如果主模型没有使用backbone，则需要在主模型的`__init__`中设置`in_channels`（默认值为3）。如果主模型使用了backbone，则不需要在主模型的`__init__`中设置`in_channels`，而是需要在backbone的`__init__`中设置`in_channels`（默认值为3）。

```python
def __init__(self,
             num_classes,
             backbone,
             backbone_indices=(2, 3),
             pp_out_channels=1024,
             bin_sizes=(1, 2, 3, 6),
             enable_auxiliary_loss=True,
             align_corners=False,
             pretrained=None):
    super().__init__()
    ...
    self.init_weight()
```

**forward 规范**

1. 逻辑尽量简洁，以组件式的调用呈现。
2. 结果需要`resize`到原图大小，按列表形式返回，第一个元素为主输出，其他为辅助输出。
3. 如果模型训练和预测时执行分支不同，使用self.training变量作为if判断，实现不同分支（示例参考PPLiteSeg模型文件）。
4. 获取Tensor的shape，建议使用`paddle.shape(x)`，不要使用`x.shape`，否则在导出预测模型的时候可能出现动转静失败的问题。
5. 组网代码统一使用Paddle API，不支持嵌入numpy操作，即不支持tensor->numpy->tensor转换。

```python
def forward(self, x):
    feat_list = self.backbone(x)
    logit_list = self.head(feat_list)
    return [
        F.interpolate(
            logit,
            paddle.shape(x)[2:],
            mode='bilinear',
            align_corners=self.align_corners) for logit in logit_list
    ]
```

**init_weight规范**

1. 调用```utils.load_entire_model```函数即可。
2. 不含 backbone 的模型可能涉及模型参数初始化，可调用`paddleseg.cvlib` 中的 `param_init`实现。

```python
# 带有 backbone 的对整个模型进行加载
def init_weight(self):
    if self.pretrained is not None:
        utils.load_entire_model(self, self.pretrained)
# 不带 backbone 的自身模型初始化
def init_weight(self):
      """Initialize the parameters of model parts."""
      for sublayer in self.sublayers():
        if isinstance(sublayer, nn.Conv2D):
          param_init.normal_init(sublayer.weight, std=0.001)
        elif isinstance(sublayer, (nn.BatchNorm, nn.SyncBatchNorm)):
          param_init.constant_init(sublayer.weight, value=1.0)
          param_init.constant_init(sublayer.bias, value=0.0)
```

#### 2.1.2 骨干网络

骨干网络Backbone的实现和主模型大体类似，具体可以参考`paddleseg/models/backbones/mobilenetv2.py`的实现。

骨干网络要求`__init__`函数输入参数必须有`in_channels`，表示输入图片的通道数，默认等于3。

骨干网络通常有多个输出特征图，比如返回分别是4、8、16和32倍下采样的特征图，便于在主模型中使用`backbone_indices`来选择使用backbone的特定特征图。

骨干网络类必须有`self.feat_channels`属性，表示所有输出特征图的通道数。

骨干网络通常有不同尺寸型号，分别通过函数进行定义，使用`@manager.BACKBONES.add_component`注册，示例如下。

```python
@manager.BACKBONES.add_component
def MobileNetV2_x0_25(**kwargs):
    model = MobileNetV2(scale=0.25, **kwargs)
    return model

@manager.BACKBONES.add_component
def MobileNetV2_x0_5(**kwargs):
    model = MobileNetV2(scale=0.5, **kwargs)
    return model
```

#### 2.1.3 分割头

目前`PaddleSeg`里面的模型只有单分割头模型，所以分割头模块直接以主模型名+Head来命名。注释规范与主模型保持一致。

```python
class PSPNetHead(nn.Layer):
```

如果是轻量级分割模型，没有`backbone`，可以看做是只有分割头的模型，那么为了简洁可以不用写Head，而把逻辑直接写在主模型部分中。

#### 2.1.4 辅助模块

除了主模型，和分割头之外的代码段都称为辅助模块。目前`PaddleSeg`已经提供了常见的辅助模块，例如`SyncBN, ConvBNReLU, FCN (AuxLayer), PPModule, ASPP, AttentionBlock`等等，详细查看```paddleseg/models/layers```。
1. **必须**优先使用`PaddleSeg`内置辅助模块；
2. 若内置辅助模块不满足需求，可以自定义模块；
3. 自定义模块保持良好的风格，以及关键模块的注释；
4. 在PR中说明，除了主模块和分割头之外，自定义模块的class名。

---
开发完模型后，在```paddleseg/models/__init__.py```中添加导入信息。若没有其他`loss`的添加，就完成了一个模型的开发。
```python
from .pspnet import *
```

### 2.2 Loss 开发规范

损失开发的规范以`paddleseg/models/losses/cross_entropy_loss.py`为例:

损失声明规范：
1. 在损失头使用`manager`装饰器；
2. 继承`nn.Layer`；
3. 添加英文注释：
   1. 损失含义：类主要做什么，损失的表达式是什么，主要针对的优化点是什么（可选）
   2. 损失参数：损失的参数比较灵活，一般可以指定权重，`ignore_index` 等
4. 必须支持`ignore_index`来设置忽略的label数值。

```python
@manager.LOSSES.add_component
class CrossEntropyLoss(nn.Layer):
    """
    Implements the cross entropy loss function.

    Args:
        weight (tuple|list|ndarray|Tensor, optional): A manual rescaling weight
            given to each class. Its length must be equal to the number of classes.
            Default ``None``.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        top_k_percent_pixels (float, optional): the value lies in [0.0, 1.0]. When its value < 1.0, only compute the loss for
            the top k percent pixels (e.g., the top 20% pixels). This is useful for hard pixel mining. Default ``1.0``.
        data_format (str, optional): The tensor format to use, 'NCHW' or 'NHWC'. Default ``'NCHW'``.
    """
```



### 2.3 Dataset 开发规范

不推荐大家直接在`paddleseg/dataset/`目录下新增数据集Class，来实现支持新的数据集。建议大家参考[准备自定义数据集文档](../../data/marker/marker_cn.md)，将数据集整理成PaddleSeg推荐的格式，基于txt文件来配置`DataSet`。

数据集开发的规范以`paddleseg/dataset/cityscapes.py`为例，文件中仅声明一个和数据集名字一致的类。

建立新的数据集文件，则在`paddleseg/dataset`中创建对应数据集名字的文件。


#### 2.3.1 数据集声明规范

1. 在类头部添加装饰器；

   ```python
   @manager.DATASETS.add_component
   ```

2. 类方法继承```Dataset```类；

3. 文档部分描述数据集来源，数据集结构，还有参数含义等。

```python
from paddleseg.dataset import Dataset

@manager.DATASETS.add_component
class Cityscapes(Dataset):
      """
    Cityscapes dataset `https://www.cityscapes-dataset.com/`.
    The folder structure is as follow:
        cityscapes
        |
        |--leftImg8bit
        |  |--train
        |  |--val
        |  |--test
        |
        |--gtFine
        |  |--train
        |  |--val
        |  |--test
    Make sure there are **labelTrainIds.png in gtFine directory. If not, please run the conver_cityscapes.py in tools.
    Args:
        transforms (list): Transforms for image.
        dataset_root (str): Cityscapes dataset directory.
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        edge (bool, optional): Whether to compute edge while training. Default: False
    """
```

#### 2.3.2 __init__规范

1. ```__init__```中参数全部显式写出，**不能**包括变长参数比如:```*args, **kwargs```；
2. ```super().__init__()```保持空参数；
3. 参数顺序和上述参数顺序一致；
4. 通过在在```__init__```方法中建立 ```self.file_list```，之后就根据其中元素的路径读取对应图片。

## 3 导出和测试预测模型

开发模型，我们不仅要关注模型精度的正确性，还需要检查模型导出和预测部署的正确性。只有模型可以顺利部署，才算真正开发完成一个模型。

### 3.1 导出预测模型

开发模型是使用PaddlePaddle的动态图模式，我们需要将动态图的模型导出为静态图的预测模型，实现更快的部署推理速度。

将动态图的模型导出为静态图的预测模型，使用的是动转静技术，此处不展开介绍，具体说明请参考[文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/04_dygraph_to_static/index_cn.html)。

请参考[文档](../../model_export_cn.md)导出静态图预测模型。如果没有报错，静态图的预测模型会保存到指定目录。如果报错，根据log修改组网代码，再次导出。

### 3.2 测试预测模型

请参考[文档](../../deployment/inference/python_inference_cn.md)，在X86 CPU或者NV GPU上使用Paddle Inference Python API加载导出的预测模型，读取单/多张图片进行测试，查看分割结果图片是否正确。

## 4 新增模型的 PR checklist

* 参考[代码提交规范](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/docs/pr/pr/pr.md)，完成代码提交前的准备，包含拉取最新内容、切换分支等。
* 在```configs```目录下有以模型名命名的子目录，比如```pspnet```，其中包含模型yml配置文件和`README.md`，详细参考[示例](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/configs/pspnet)。
* 模型yml配置文件的文件命名方式为`模型名+backbone+out_stride+数据集+训练分辨率+训练单卡iters.yml`，不含部分就略去，详细参考[配置项文档](../../design/use/use_cn.md)。
* `README.md`中模型原文文献的reference风格采用Chicago，即全部作者名，详细参考[示例](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/configs/pspnet/README.md)。
* `README.md`中，提供至少一个数据集上的训练和测试结果，格式如下表格
    * `Resolution`表示训练数据集预处理时crop的图像尺寸。
    * `mIoU、mIoU(flip)、mIoU(ms+flip)`是模型在验证集上的评估精度。`flip`表示测试使用水平翻转；`ms` 表示`multi-scale`，即使用三种`scale` [0.75, 1.0, 1.25]进行测试；`ms+flip`表示使用这两种数据处理方式进行测试。模型评估的详细使用方法，请参考[模型评估](../../evaluation/evaluate_cn.md)。
    * 提供的下载链接包括三个部分：训练好的模型参数，训练日志，训练vdl。

    | Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    ||||||||[model]() \| [log]() \| [vdl]()|

* 新增的代码文件，参考本文档前面介绍的`新增文件的开发规范`和`可拓展模块的开发规范`进行自查和改正，参考`导出和测试预测模型`完成测试并在PR中反馈给Reviewer。

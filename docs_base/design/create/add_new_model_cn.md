简体中文 | [English](add_new_model.md)
# 添加组件

> PaddleSeg 提供了五种类型的可扩展组件，即 MODELS、LOSSES、TRANSFORMS、BACKBONES、DATASETS。

> PaddleSeg使用基于面向对象的设计思想，在创造你自己的模型时，请以 Python class 的形式编写。

## 创建自定义分割模型

> 如果你打算设计一个自定义分割模型，例如在 newnet.py 中实现 NewNet 类（你可以为你的模型起任何名字，但不要与已有模型的名字重复）:

```python
import paddle.nn as nn
from paddleseg.cvlibs import manager

@manager.MODELS.add_component
class NewNet(nn.Layer):
    def __init__(self, param1, param2, param3):
        pass
    def forward(self, x):
        pass
```

* **步骤 1**: 将 newnet.py 文件放置在目录 paddleseg/models/ 下。

* **步骤 2**: 在你的自定义模型类的上方添加一个Python装饰器 ``@manager.MODELS.add_component``。manager 是一个组件容器，包括 MODELS、BACKBONES、DATASETS、TRANSFORMS、LOSSES。当你添加了这个装饰器并在训练时合理的指定参数，PaddleSeg就可以自动将你实现的模块添加到训练配置中，体现了低耦合的设计思想。

* **步骤 3**: 在 paddleseg/models/\_\_init\_\_.py 中导入你的自定义分割模型类，如下所示:
```python
from .backbones import * # 已经实现的骨干网络类
from .losses import * # 已经实现的损失函数类

from .ann import * #目前已经实现的21种分割模型。你将按照同样的规则添加自己的自定义分割模型。
from .bisenet import *
from .danet import *
from .deeplab import *
from .fast_scnn import *
from .fcn import *
from .gcnet import *
from .ocrnet import *
from .pspnet import *
from .gscnn import GSCNN
from .unet import UNet
from .hardnet import HarDNet
from .u2net import U2Net, U2Netp
from .attention_unet import AttentionUNet
from .unet_plusplus import UNetPlusPlus
from .unet_3plus import UNet3Plus
from .decoupled_segnet import DecoupledSegNet
from .emanet import *
from .isanet import *
from .dnlnet import *
from .sfnet import *
from .shufflenet_slim import ShuffleNetV2 # 以上导入请不要改动，否则会导致一些模型不可用

from .newnet import NewNet # 请在这里添加你自己实现的分割模型类
```
- 注意：如果仅实现了自己的类，而不在 models 的构造函数中导入你的类，PaddleSeg将无法识别你添加的模型！

* **步骤 4**: 在 yaml 文件中将 type 参数指定为你所创建的分割模型的名称（该参数必须与 newnet.py 中的 NewNet 类名保持一致，请不要误输为python文件的名称）。

> 另外，请你记住所创建的 yaml 文件的完整路径，以便后续在为 train.py 设置模型配置参数时使用你想要的配置。建议将模型的 yaml 文件都保存在 PaddleSeg/configs 的对应模型目录下。

```yaml
model:
  type: NewNet # 你的自定义模型名称
  param1: ...
  param2: ...
  param3: ...
```

- 注意：如果你的模型有多个输出，即损失 = 主损失 + 辅助损失，则你必须修改 yaml 文件中的相应参数，否则会抛出“ logits 的长度应该等于 loss 配置的类型数目： 2!=1。”的错误。 比如 PSPNet 有两个 loss，且都是 CrossEntropyLoss，辅助loss的权重是0.4，所以我们对 loss 的设置如下：

```yaml
loss:
  types:
    - type: CrossEntropyLoss
  coef: [1, 0.4] #为主损失和辅助损失分配不同的比重，即主损失对最终损失影响更大。
```


## 创建自定义损失函数

> 如果你打算设计一个自定义损失函数，例如在 new_loss.py 中实现 NewLoss 类（你可以为你的损失函数起任何名字，但不要与已有损失函数的名字重复）:


```python
import paddle.nn as nn
from paddleseg.cvlibs import manager

@manager.LOSSES.add_component
class NewLoss(nn.Layer):
    def __init__(self, param1, ignore_index=255):
        pass
    def forward(self, x):
        pass
```



* **步骤 1**: 将 new_loss.py 文件放置在目录 paddleseg/models/losses 下。

* **步骤 2**: 在你的自定义损失函数类的上方添加一个Python装饰器 ``@manager.LOSSES.add_component``。

* **步骤 3**: 在 paddleseg/models/losses/\_\_init\_\_.py 中导入你的自定义损失函数类，如下所示：

```python
from .mixed_loss import MixedLoss
from .cross_entropy_loss import CrossEntropyLoss
from .binary_cross_entropy_loss import BCELoss
from .lovasz_loss import LovaszSoftmaxLoss, LovaszHingeLoss
from .gscnn_dual_task_loss import DualTaskLoss
from .edge_attention_loss import EdgeAttentionLoss
from .bootstrapped_cross_entropy import BootstrappedCrossEntropyLoss
from .dice_loss import DiceLoss
from .ohem_cross_entropy_loss import OhemCrossEntropyLoss
from .decoupledsegnet_relax_boundary_loss import RelaxBoundaryLoss
from .ohem_edge_attention_loss import OhemEdgeAttentionLoss
from .l1_loss import L1Loss
from .mean_square_error_loss import MSELoss # 以上导入请不要改动，否则会导致一些损失函数不可用

from .new_loss import NewLoss # 请在这里添加你自己实现的损失函数类
```

* **步骤 4**: 在 yaml 文件中将 type 参数指定为你所创建的损失函数的名称。

```yaml
loss:
  types:
    - type: NewLoss # 你的自定义损失函数名称
      param1: ...
  coef: [1]
```

## 创建自定义数据变换（数据增强）


> 如果你打算设计一个自定义数据变换（数据增强），例如在 transforms.py 中新实现一个 NewTrans 类:

```python

@manager.TRANSFORMS.add_component
class NewTrans(nn.Layer):
    def __init__(self, param1):
        pass
    def __call__(self, im, label=None):

        ...

        if label is None:
            return (im, )
        else:
            return (im, label)
```


* **步骤 1**: 在 paddleseg/transforms/transforms.py 文件中定义 NewTrans 类。

* **步骤 2**: 在你的 transform 类的上方添加一个Python装饰器 ``@manager.TRANSFORMS.add_component``。这样就可以了。
- 请注意，不再需要将该类导入到transforms的构造函数中了。在PaddleSeg中，transform组件把所有类都集成在一个文件里。你可以查看PaddleSeg/paddleseg/transforms/\_\_init\_\_.py，看看其文件内容与上文提到的 paddleseg/models/\_\_init\_\_.py 与 paddleseg/models/losses/\_\_init\_\_.py 的文件内容有何不同。

```python
from .transforms import *
from . import functional

# 可以看到，PaddleSeg/paddleseg/transforms/\_\_init\_\_.py 文件中，以from .transforms import *导入所有已有的数据变换策略。

# 因此在你的自定义 transform 类写好后，在类对象创建过程中，它会被自动添加进来。
```

* **步骤 3**: 在 yaml 文件中将 type 参数指定为你所创建的数据变换（数据增强）的名称:

```yaml
train_dataset:
  transforms:
    - type: NewTrans # 你的自定义数据变换名称
      param1: ...
```

> 注意：为了更好的可读性，请在 paddleseg/transforms/functional.py 中实现详细的转换函数。


## 创建自定义骨干网络


> 如果你打算设计一个自定义骨干网络，例如在 new_backbone.py 中实现 NewBackbone 类（你可以为你的骨干网络起任何名字，但不要与已有骨干网络的名字重复）:

```python
import paddle.nn as nn
from paddleseg.cvlibs import manager

@manager.BACKBONES.add_component
class NewBackbone(nn.Layer):
    def __init__(self, param1):
        pass
    def forward(self, x):
        pass
```



* **步骤 1**: 将 new_backbone.py 文件放置在目录 paddleseg/models/backbones 下。

* **步骤 2**: 在你的自定义骨干网络类的上方添加一个Python装饰器 ``@manager.BACKBONES.add_component``。

* **步骤 3**: 在 paddleseg/models/backbones/\_\_init\_\_.py 中导入你的自定义骨干网络类，如下所示：
```python
# 目前支持4种骨干网络
from .hrnet import *
from .resnet_vd import *
from .xception_deeplab import *
from .mobilenetv3 import * # 以上导入请不要改动，否则会导致一些骨干网络不可用

from .new_backbone import NewBackbone # 请在这里添加你自己实现的骨干网络类
```

* **步骤 4**: 在 yaml 文件中将 type 参数指定为你所创建的骨干网络的名称。

```yaml
model:
  backbone:
    type: NewBackbone # 你的自定义骨干网络名称
    param1: ...
```

## 创建自定义数据集


> 如果你打算设计一个自定义数据集，例如在 new_data.py 中实现 NewData 类:

```python
from paddleseg..dataset import Dataset
from paddleseg.cvlibs import manager

@manager.DATASETS.add_component
class NewData(Dataset):
    def __init__(self,
                 dataset_root=None,
                 transforms=None,
                 mode='train'):
        pass
```


* **步骤 1**: 将 new_data.py  文件放置在目录 paddleseg/datasets 下。

* **步骤 2**: 在你的自定义数据集类的上方添加一个Python装饰器 ``@manager.DATASETS.add_component``。

* **步骤 3**: 在 paddleseg/datasets/\_\_init\_\_.py 中导入你的自定义数据集类，如下所示：
```python
from .dataset import Dataset
from .cityscapes import Cityscapes
from .voc import PascalVOC
from .ade import ADE20K
from .optic_disc_seg import OpticDiscSeg
from .pascal_context import PascalContext
from .mini_deep_globe_road_extraction import MiniDeepGlobeRoadExtraction # 以上导入请不要改动，否则会导致一些数据集不可用

from .new_data import NewData # 请在这里添加你自己实现的数据集类
```
* **步骤 4**: 在 yaml 文件中将 type 参数指定为你所创建的数据集的名称。

```yaml
train_dataset:
  type: NewData # 你的自定义数据集名称
  dataset_root: ...
  mode: train
```


# 举例
> 假设我们已经按照以上步骤编写好了五个自定义组件：MODELS（NewNet）、LOSSES（NewLoss）、TRANSFORMS（NewTrans）、BACKBONES（NewBackbone）、DATASETS（NewData）。
> 假设我们要为自定义模型的训练编写 yaml 文件，以设定本次用到的参数。此处我们主要关心对自定义组件参数的配置，其他参数（如优化器）不过多介绍，建议沿用参考配置。如:
```yaml
batch_size: 4 # 设定迭代一次送入网络的图片数量。一般来说，你所使用机器的显存越大，可以调高batch_size的值。

iters: 10000 # 迭代次数

model:
  type: NewNet # 自定义模型类的名称
  backbone:
    type: NewBackbone # 自定义骨干网络类的名称
    pretrained: Null # 如果你实现训练过骨干网络的参数，请指定其存放路径
  num_classes: 5 # 标注图中像素类别个数。你的模型应该是根据具体的分割任务设计的，因此你知道该任务下像素类别个数
  pretrained: Null # 如果你有网络的与预训练参数，请指定其存放路径
  backbone_indices: [-1]

loss:
  types:
    - type: NewLoss # 自定义损失函数类的名称
  coef: [1] # 若使用多种loss，该列表长度与loss数目保持一致


train_dataset:
  type: NewData # 自定义数据集类的名称
  dataset_root: data/custom_data # 请参考README文档，按其推荐的整理结构组织分割任务所需要的文件，将其放在相应的项目路径下。推荐放在data/下。
  transforms:
    - type: NewTrans # 自定义数据转换（数据增强）类的名称
      custom_factor: 0.5
  mode: train # 对训练集设定训练模式


val_dataset:
  type: NewData
  dataset_root: data/custom_data
  transforms:
    - type: Normalize
  mode: val # 对验证集设定验证模式

optimizer: # 优化器设置
  type: sgd
  momentum: 0.9
  weight_decay: 4.0e-5

lr_scheduler: # 学习率的设置
  type: PolynomialDecay
  learning_rate: 0.01
  power: 0.9
  end_lr: 0
```

假设我们将上面的 yaml 文件保存为 PaddleSeg/configs/custom_configs/NewNet_NewLoss_NewTrans_NewBackbone_NewData.yml，请先切换到PaddleSeg目录下后，运行以下命令：
```
python tools/train.py \
       --config configs/custom_configs/NewNet_NewLoss_NewTrans_NewBackbone_NewData.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output
```

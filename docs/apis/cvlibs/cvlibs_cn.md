简体中文 | [English](cvlibs.md)
# paddleseg.cvlibs

- [ComponentManager](#ComponentManager)
- [constant_init](#constant_init)
- [Config](#Config)


## [ComponentManager](../../../paddleseg/cvlibs/manager.py)
```python
class paddleseg.cvlibs.manager.ComponentManager(name)
```
> 实现一个manager类，以正确地向模型添加新组件。
> 组件以类或函数的类型被添加。 


### 参数
* **name** (str): 组件名。

### 示例 1

```python
from paddleseg.cvlibs.manager import ComponentManager

model_manager = ComponentManager()

class AlexNet: ...
class ResNet: ...

model_manager.add_component(AlexNet)
model_manager.add_component(ResNet)

# 或传入一个可迭代的序列
model_manager.add_component([AlexNet, ResNet])
print(model_manager.components_dict)
# {'AlexNet': <class '__main__.AlexNet'>, 'ResNet': <class '__main__.ResNet'>}
```

### 示例 2:

```python
# 或者用一种更简单的方法，将它作为 Python 装饰器，同时将它添加到类声明之上。
from paddleseg.cvlibs.manager import ComponentManager

model_manager = ComponentManager()

@model_manager.add_component
class AlexNet: ...

@model_manager.add_component
class ResNet: ...

print(model_manager.components_dict)
# {'AlexNet': <class '__main__.AlexNet'>, 'ResNet': <class '__main__.ResNet'>}
```

```python
add_component(components)
```
> 向相关manager添加组件（≥1个）

### 参数
* **components** (function|class|list|tuple): 支持四种类型的组件。
### 返回值
* **components** (function|class|list|tuple): 与输入组件相同。

## [constant_init](../../../paddleseg/cvlibs/param_init.py)

```python
constant_init(param, **kwargs):
```
> 用常量初始化 `param` 

### 参数
* **param** (Tensor): 需要被初始化的张量。

### 示例

```python
from paddleseg.cvlibs import param_init
import paddle.nn as nn

linear = nn.Linear(2, 4)
param_init.constant_init(linear.weight, value=2.0)
print(linear.weight.numpy())
# 结果为[[2. 2. 2. 2.], [2. 2. 2. 2.]]
```

### [normal_init](../../../paddleseg/cvlibs/param_init.py)
```python
normal_init(param, **kwargs)
```
> 用正态分布初始化 `param` 

### 参数
* **param** (Tensor): Tensor that needs to be initialized.

### 示例

```python
from paddleseg.cvlibs import param_init
import paddle.nn as nn

linear = nn.Linear(2, 4)
param_init.normal_init(linear.weight, loc=0.0, scale=1.0)
```

### [kaiming_normal_init](../../../paddleseg/cvlibs/param_init.py)
```python
kaiming_normal_init(param, **kwargs):
```

> 用Kaiming Normal initialization对输入张量进行初始化。

> 本函数是对论文`Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <https://arxiv.org/abs/1502.01852>`中对`param` 的初始化方法的实现。该论文作者为Kaiming He, Xiangyu Zhang, Shaoqing Ren 和Jian Sun。
    
> 由于其特别考虑到整流器的非线性特征，这是一种具有鲁棒性的初始化方法。

> 在均匀分布的情况下，范围是 [-x, x]，其中
    <a href="https://www.codecogs.com/eqnedit.php?latex=x&space;=&space;\sqrt{\frac{6.0}{fan_in}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x&space;=&space;\sqrt{\frac{6.0}{fan_in}}" title="x = \sqrt{\frac{6.0}{fan_in}}" /></a>

> 在正态分布的情况下，均值为 0，标准差为
    <a href="https://www.codecogs.com/eqnedit.php?latex=\sqrt{\frac{2.0}{fan_in}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sqrt{\frac{2.0}{fan_in}}" title="\sqrt{\frac{2.0}{fan_in}}" /></a>

### 参数
* **param** (Tensor): 需要被初始化的张量。

### 示例 

```python
from paddleseg.cvlibs import param_init
import paddle.nn as nn

linear = nn.Linear(2, 4)
# uniform 用于决定是使用均匀分布还是正态分布
param_init.kaiming_normal_init(linear.weight)
```

## [Config](../../../paddleseg/cvlibs/config.py)
```python
class Config(
    path, 
    learning_rate, 
    batch_size, 
    iters
)
```

* 对训练配置文件的解析。仅支持 yaml/yml 类型的配置文件。

* 可在配置文件中配置以下超参数：
> **batch_size**: 对于每个GPU，指定其每轮训练要处理样本的个数。

> **iters**: 总训练轮数。

> **train_dataset**: 对一个训练数据集的配置包括设置 type/data_root/transforms/mode。对于数据类型，请参考paddleseg.datasets。对于特殊变换，请参考paddleseg.transforms.transforms。

> **val_dataset**: 对一个验证数据集的配置包括设置 type/data_root/transforms/mode。
        
> **optimizer**: 配置一个优化器。但现在PaddleSeg仅支持在配置文件中设置带momentum的sgd。此外，weight_decay （权重衰减）可以设置为正则化。

> **learning_rate**: 一个学习率配置。如果配置了学习率，learning _rate 值将作为初始学习率，其中使用配置文件仅支持poly衰减。 此外，衰减率和 end_lr 是通过实验调整的。


> **loss**: 一个损失函数的配置。可以混合使用多种损失函数。损失函数类型的顺序应与seg模型的输出保持一致，其中 coef 项表示相应损失的权重。 注意，coef 的个数必须和模型输出的个数相同，如果输出中使用相同的 loss 类型，则在 type 列表中只能出现一种 loss 类型，否则 loss 类型的个数必须与 coef 一致。

> **model**: 对一个model的配置信息应包含 模型类别/骨干网络 以及超参数。对于模型类别，请参考paddleseg.models。对于骨干网络，请参考paddleseg.models.backbones。

### 参数
* **path** (str) : 配置文件的路径, 仅支持 yaml 格式。

### 示例

```python
from paddleseg.cvlibs.config import Config

# 使用 yaml 文件路径创建一个 cfg 对象。
cfg = Config(yaml_cfg_path)

# 在使用该对象的属性时，解析参数。（比如使用 train_dataset 属性）
train_dataset = cfg.train_dataset

#模型的参数应该在数据集之后解析，因为模型的构建器要使用数据集中的一些属性。
model = cfg.model
...
```

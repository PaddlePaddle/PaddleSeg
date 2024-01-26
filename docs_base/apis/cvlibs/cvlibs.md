English | [简体中文](cvlibs_cn.md)
# paddleseg.cvlibs

- [ComponentManager](#ComponentManager)
- [constant_init](#constant_init)
- [Config](#Config)


## [ComponentManager](../../../paddleseg/cvlibs/manager.py)
```python
class paddleseg.cvlibs.manager.ComponentManager(name)
```
> Implement a Manager class to correctly add new components to the model.
> Components are added as classes or functions.

### Args
* **name** (str): The name of component.

### Examples 1

```python
from paddleseg.cvlibs.manager import ComponentManager

model_manager = ComponentManager()

class AlexNet: ...
class ResNet: ...

model_manager.add_component(AlexNet)
model_manager.add_component(ResNet)

# Or pass a sequence alliteratively:
model_manager.add_component([AlexNet, ResNet])
print(model_manager.components_dict)
# {'AlexNet': <class '__main__.AlexNet'>, 'ResNet': <class '__main__.ResNet'>}
```

### Examples 2:

```python
# Or an easier way, using it as a Python decorator, while just add it above the class declaration.
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
> Add component(s) into the corresponding manager.

### Args
* **components** (function|class|list|tuple): Support four types of components.
### Returns
* **components** (function|class|list|tuple): Same with input components.


## [constant_init](../../../paddleseg/cvlibs/param_init.py)

```python
constant_init(param, **kwargs):
```
> Initialize the `param` with constants.

### Args
* **param** (Tensor): Tensor that needs to be initialized.

### Examples

```python
from paddleseg.cvlibs import param_init
import paddle.nn as nn

linear = nn.Linear(2, 4)
param_init.constant_init(linear.weight, value=2.0)
print(linear.weight.numpy())
# result is [[2. 2. 2. 2.], [2. 2. 2. 2.]]
```

### [normal_init](../../../paddleseg/cvlibs/param_init.py)
```python
normal_init(param, **kwargs)
```

    Initialize the `param` with a Normal distribution.

### Args
* **param** (Tensor): Tensor that needs to be initialized.

### Examples

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

> Initialize the input tensor with Kaiming Normal initialization.

> This function implements the `param` initialization from the paper
    `Delving Deep into Rectifiers: Surpassing Human-Level Performance on
    ImageNet Classification <https://arxiv.org/abs/1502.01852>`
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. This is a
    robust initialization method that particularly considers the rectifier
    nonlinearities.
> In case of Uniform distribution, the range is [-x, x], where
    <a href="https://www.codecogs.com/eqnedit.php?latex=x&space;=&space;\sqrt{\frac{6.0}{fan_in}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x&space;=&space;\sqrt{\frac{6.0}{fan_in}}" title="x = \sqrt{\frac{6.0}{fan_in}}" /></a>

> In case of Normal distribution, the mean is 0 and the standard deviation
    is
    <a href="https://www.codecogs.com/eqnedit.php?latex=\sqrt{\frac{2.0}{fan_in}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sqrt{\frac{2.0}{fan_in}}" title="\sqrt{\frac{2.0}{fan_in}}" /></a>

### Args
* **param** (Tensor): Tensor that needs to be initialized.

### Examples

```python
from paddleseg.cvlibs import param_init
import paddle.nn as nn

linear = nn.Linear(2, 4)
# uniform is used to decide whether to use uniform or normal distribution
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
* Training configuration parsing. The only yaml/yml file is supported.

* The following hyper-parameters are available in the config file:
> **batch_size**: The number of samples per gpu.
> **iters**: The total training steps.
> **train_dataset**: A training data config including type/data_root/transforms/mode.For data type, please refer to paddleseg.datasets.For specific transforms, please refer to paddleseg.transforms.transforms.

> **val_dataset**: A validation data config including type/data_root/transforms/mode.

> **optimizer**: A optimizer config, but currently PaddleSeg only supports sgd with momentum in config file.In addition, weight_decay could be set as a regularization.

> **learning_rate**: A learning rate config. If decay is configured, learning _rate value is the starting learning rate,where only poly decay is supported using the config file. In addition, decay power and end_lr are tuned experimentally.

> **loss**: A loss config. Multi-loss config is available. The loss type order is consistent with the seg model outputs,where the coef term indicates the weight of corresponding loss. Note that the number of coef must be the same as the number of model outputs, and there could be only one loss type if using the same loss type among the outputs, otherwise the number of loss type must be consistent with coef.

> **model**: A model config including type/backbone and model-dependent arguments.For model type, please refer to paddleseg.models.For backbone, please refer to paddleseg.models.backbones.

### Args
* **path** (str) : The path of config file, supports yaml format only.

### Examples

```python
from paddleseg.cvlibs.config import Config

# Create a cfg object with yaml file path.
cfg = Config(yaml_cfg_path)

# Parsing the argument when its property is used.
train_dataset = cfg.train_dataset

# the argument of model should be parsed after dataset,
# since the model builder uses some properties in dataset.
model = cfg.model
...
```

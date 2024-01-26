# Add New Components

PaddleSeg provides five types of extensible components, i.e. MODELS, LOSSES, TRANSFORMS, BACKBONES, DATASETS.

## A New Model

If you intent to design a customized model, e.g, NewNet in newnet.py:

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

**Step 1**: Put newnet.py under paddleseg/models/.

**Step 2**: Add @manager.MODELS.add_component above your model class, where the manager is a component container, inclduing MODELS, BACKBONES, DATASETS, TRANSFORMS, LOSSES.

**Step 3**: Import your class in paddleseg/models/\_\_init\_\_.py, like this:
```python
from .newnet import NewNet
```

**Step 4**: Specify the model name in a yaml file:

```python
model:
  type: NewNet
  param1: ...
  param2: ...
  param3: ...
```

Note: If your model has more than one output, i.e. main loss + auxiliary losses, you have to modify loss in the yaml file, otherwise it will throw out an error like "The length of logits should equal to the types of loss config: 2!=1.". For example, PSPNet has two losses, where both are CrossEntropyLoss, and the weight of auxilary loss is 0.4, thus we have the loss settings:
```python
loss:
  types:
    - type: CrossEntropyLoss
  coef: [1, 0.4]
```

## A New Loss

If you intent to implement a new loss, e.g. NewLoss in new_loss.py.

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

**Step 1**: Put new_loss.py under paddleseg/models/losses.

**Step 2**: Add @manager.LOSSES.add_component above your loss class.

**Step 3**: Import your class in paddleseg/models/losses/\_\_init\_\_.py, like this:
```python
from .new_loss import NewLoss
```

**Step 4**: Specify the loss name in a yaml file:

```python
loss:
  types:
    - type: NewLoss
      param1: ...
  coef: [1]
```

## A New Transform

If you intent to implement a new transform (data augmentation), e.g. NewTrans.

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

**Step 1**: Define the NewTrans class in paddleseg/transforms/transforms.py.

**Step 2**: Add @manager.TRANSFORMS.add_component above your transform class. That's all.

**Step 3**: Specify the transform name in a yaml file:

```python
train_dataset:
  transforms:
    - type: NewTrans
      param1: ...
```
Note: For better readability，please implement detailed transformation functions in paddleseg/transforms/functional.py.

## A New Backbone

If you intent to add a new backbone network, e.g. NewBackbone in new_backbone.py.

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

**Step 1**: Put new_backbone.py under paddleseg/models/backbones.

**Step 2**: Add @manager.BACKBONES.add_component above your backbone class.

**Step 3**: Import your class in paddleseg/models/backbones/\_\_init\_\_.py, like this:
```python
from .new_backbone import NewBackbone
```

**Step 4**: Specify the backbone name in a yaml file:

```python
model:
  backbone:
    type: NewBackbone
    param1: ...
```

## A New Dataset

If you intent to add a new dataset, e.g. NewData in new_data.py.

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

**Step 1**: Put new_data.py under paddleseg/datasets.

**Step 2**: Add @manager.DATASETS.add_component above your dataset class.

**Step 3**: Import your class in paddleseg/datasets/\_\_init\_\_.py, like this:
```python
from .new_data import NewData
```

**Step 4**: Specify the backbone name in a yaml file:

```python
train_dataset:
  type: NewData
  dataset_root: ...
  mode: train
```

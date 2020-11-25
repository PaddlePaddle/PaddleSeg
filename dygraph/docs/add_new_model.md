# Add New Components

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

Step 1: Put newnet.py under paddleseg/models/.

Step 2: Add @manager.MODELS.add_component above your model class, where the manager is a commpent container, inclduing MODELS, BACKBONES, DATASETS, TRANSFORMS, LOSSES.

Step 3: Import your class in paddleseg/models/__init__.py, like this:
```python
from .newnet import NewNet
```

Step 4: Use the model name in a yaml file:

```python
model:
  type: NewNet
  param1: ...
  param2: ...
  param3: ...
```

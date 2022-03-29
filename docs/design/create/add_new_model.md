English | [简体中文](add_new_model_cn.md)
# Add New Components

> PaddleSeg provides five types of extensible components, i.e. MODELS, LOSSES, TRANSFORMS, BACKBONES, DATASETS.

> PaddleSeg uses object-oriented design ideas. When creating your own model, please write it in the form of Python class.

## A New Model

> If you intent to design a customized model, e.g, NewNet in newnet.py (you are allowed to give your model any name, but don't conflict with an existing model name):

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

* **Step 1**: Put newnet.py under paddleseg/models/.

* **Step 2**: Add @manager.MODELS.add_component above your model class, where the manager is a component container, inclduing MODELS, BACKBONES, DATASETS, TRANSFORMS, LOSSES.When you add this decorator and specify the parameters reasonably during training, PaddleSeg can automatically add the modules you implement to the training configuration, reflecting the design idea of low coupling.

* **Step 3**: Import your class in paddleseg/models/\_\_init\_\_.py, like this:
```python
from .backbones import * # Backbone network classes that have been implemented
from .losses import * # Losses classes that have been implemented

from .ann import * #Currently 21 segmentation models have been implemented. You will add your own custom segmentation model following the same rules.
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
from .shufflenet_slim import ShuffleNetV2 # Please do not change the above import, otherwise it will cause some models to be unavailable

from .newnet import NewNet # Please add your own segmentation model class here.
```
- Note: If you only implement your own class without importing your class in the models constructor, PaddleSeg will not be able to recognize the model you added!

* **Step 4**: Specify the model name in a yaml file:(This parameter must be consistent with the NewNet class name in newnet.py, please do not enter the name of the python file by mistake).

> In addition, please remember the full path of the created yaml file so that you can use the configuration you want when setting the model configuration parameters for train.py later. It is recommended to save the yaml files of the model in the corresponding model directory of PaddleSeg/configs.

```yaml
model:
  type: NewNet # Your custom model name.
  param1: ...
  param2: ...
  param3: ...
```

- Note: If your model has more than one output, i.e. main loss + auxiliary losses, you have to modify loss in the yaml file, otherwise it will throw out an error like "The length of logits should equal to the types of loss config: 2!=1.". For example, PSPNet has two losses, where both are CrossEntropyLoss, and the weight of auxilary loss is 0.4, thus we have the loss settings:

```yaml
loss:
  types:
    - type: CrossEntropyLoss
  coef: [1, 0.4] # The main loss and auxiliary loss are assigned different proportions, that is, the main loss has a greater impact on the final loss.
```


## A New Loss

> If you intent to implement a new loss, e.g. NewLoss in new_loss.py (you are allowed to give your model any name, but don't conflict with an existing loss name):


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



* **Step 1**: Put new_loss.py under paddleseg/models/losses.

* **Step 2**: Add @manager.LOSSES.add_component above your loss class.

* **Step 3**: Import your class in paddleseg/models/losses/\_\_init\_\_.py, like this:

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
from .mean_square_error_loss import MSELoss # Please do not change the above import, otherwise it will cause some loss functions to be unavailable.

from .new_loss import NewLoss # Please add your own loss function class here.
```

* **Step 4**: Specify the loss name in a yaml file:

```yaml
loss:
  types:
    - type: NewLoss # The name of your custom loss function
      param1: ...
  coef: [1]
```

## A New Transform(Data Augmentation)


> If you intent to implement a new transform (data augmentation), e.g. NewTrans (you are allowed to give your model any name, but don't conflict with an existing transform name):

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


* **Step 1**: Define the NewTrans class in paddleseg/transforms/transforms.py.

* **Step 2**: Add @manager.TRANSFORMS.add_component above your transform class. That's all.
- Please note that it is no longer necessary to import this class into the constructor of transforms. In PaddleSeg, the transform component integrates all classes in one file. You can check PaddleSeg/paddleseg/transforms/\_\_init\_\_.py to see the file content and the above mentioned paddleseg/models/\_\_init\_\_.py and paddleseg/models/ What is the difference in the file content of losses/\_\_init\_\_.py.

```python
from .transforms import *
from . import functional

# As you can see, in the PaddleSeg/paddleseg/transforms/\_\_init\_\_.py file, import all existing data transformation strategies with from .transforms import *.

# Therefore, after your custom transform class is written, it will be automatically added during the creation of the class object.
```

* **Step 3**: Specify the type parameter in the yaml file as the name of the data transformation (data enhancement) you created:

```yaml
train_dataset:
  transforms:
    - type: NewTrans # Your custom data transformation name
      param1: ...
```

> Note: For better readability, please implement detailed conversion functions in paddleseg/transforms/functional.py.


## A New Backbone


> If you intent to add a new backbone network, e.g. NewBackbone in new_backbone.py（you are allowed to give your model any name, but don't conflict with an existing backbone name）:

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



* **Step 1**: Put new_backbone.py under paddleseg/models/backbones.

* **Step 2**: Add @manager.BACKBONES.add_component above your backbone class.

* **Step 3**: Import your class in paddleseg/models/backbones/\_\_init\_\_.py, like this:
```python
# Currently supports 4 types of backbone networks.
from .hrnet import *
from .resnet_vd import *
from .xception_deeplab import *
from .mobilenetv3 import * # Please do not modify the above import, otherwise it will cause some backbone networks to be unavailable.

from .new_backbone import NewBackbone # Please add your own backbone network class here.
```

* **Step 4**: Specify the backbone name in a yaml file:

```yaml
model:
  backbone:
    type: NewBackbone # Your custom backbone network name.
    param1: ...
```

## A New Dataset


> If you intent to add a new dataset, e.g. NewData in new_data.py:

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


* **Step 1**: Put new_data.py under paddleseg/datasets.

* **Step 2**: Add @manager.DATASETS.add_component above your dataset class.

* **Step 3**: Import your class in paddleseg/datasets/\_\_init\_\_.py, like this:
```python
from .dataset import Dataset
from .cityscapes import Cityscapes
from .voc import PascalVOC
from .ade import ADE20K
from .optic_disc_seg import OpticDiscSeg
from .pascal_context import PascalContext
from .mini_deep_globe_road_extraction import MiniDeepGlobeRoadExtraction # Please do not change the above import, otherwise it will cause some data sets to be unavailable.

from .new_data import NewData # Please add your own implementation of the dataset class here.
```
* **Step 4**: Specify the backbone name in a yaml file:

```yaml
train_dataset:
  type: NewData # Your custom data set name.
  dataset_root: ...
  mode: train
```


# Example
> Suppose we have written five custom components according to the above Steps: MODELS (NewNet), LOSSES (NewLoss), TRANSFORMS (NewTrans), BACKBONES (NewBackbone), DATASETS (NewData).
> Suppose we want to write a yaml file for the training of a custom model to set the parameters used this time. Here we are mainly concerned with the configuration of custom component parameters, other parameters (such as optimizers) are not introduced too much, and it is recommended to use the reference configuration. Such as:
```yaml
batch_size: 4 # Set the number of pictures sent to the network at one iteration. Generally speaking, the larger the video memory of the machine you are using, the higher the batch_size value.

iters: 10000 # Number of iterations.

model:
  type: NewNet # The name of the custom model class.
  backbone:
    type: NewBackbone # Customize the name of the backbone network class.
    pretrained: Null # If you implement the parameters of the trained backbone network, please specify its storage path.
  num_classes: 5 # Label the number of pixel categories in the image. Your model should be designed according to the specific segmentation task, so you know the number of pixel categories under the task.
  pretrained: Null # If you have network and pre-training parameters, please specify the storage path.
  backbone_indices: [-1]

loss:
  types:
    - type: NewLoss # The name of the custom loss function class
  coef: [1] # If multiple losses are used, the length of the list is consistent with the number of losses.


train_dataset:
  type: NewData # The name of the custom dataset class
  dataset_root: data/custom_data # Please refer to the README file, organize the files needed for the segmentation task according to its recommended organization structure, and place them in the corresponding project path. It is recommended to put it under data/
  transforms:
    - type: NewTrans # The name of the custom data conversion (data enhancement) class.
      custom_factor: 0.5
  mode: train # Set training mode for training set.


val_dataset:
  type: NewData
  dataset_root: data/custom_data
  transforms:
    - type: Normalize
  mode: val # Set the verification mode for the verification set.

optimizer: # Optimizer settings.
  type: sgd
  momentum: 0.9
  weight_decay: 4.0e-5

lr_scheduler: # Learning rate setting.
  type: PolynomialDecay
  learning_rate: 0.01
  power: 0.9
  end_lr: 0
```

Suppose we save the above yaml file as PaddleSeg/configs/custom_configs/NewNet_NewLoss_NewTrans_NewBackbone_NewData.yml, please switch to the PaddleSeg directory and run the following command:
```
python train.py \
       --config configs/custom_configs/NewNet_NewLoss_NewTrans_NewBackbone_NewData.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output
```

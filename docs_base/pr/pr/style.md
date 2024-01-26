English | [简体中文](style_cn.md)
# PaddleSeg model development specification

Model specifications includes following parts: new file self-inspection, expandable module specifications, PR checklist, and export and infer prediction models.


## 1 New file self-inspection

The newly add files need to be self checked, this mainly includes copyright, import, and coding checklist.

### 1.1 copyright

After creating an empty file `pspnet.py`, add the following copyright at the top of the file. Each new file in PaddleSeg needs to add corresponding copyright information. Note: The year should be rewritten if it is not correct.

```python
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

### 1.2 import

This part import the packages the model needs. Three types of package should be imported in the following order for each new file:
1. Python package;
2. The third party packages, which include the packages you install through `pip` or `conda install`;
3. Package in PaddleSeg.

The following code is the import example.

Note:
1. Blank lines should be inserted between different types of package.

2. Unused packages should be deleted.
3. If the import code length is too long/short, arrange them in increasing order.

```python
import os

import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils
```

### 1.3 Self-check coding checklist

This part explains the specifications that need to be paid attention to in python coding. Most of specifications will be checked and refined by pre-commit. For more information, please refer to [Google Programming Guidelines](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/) .

- [ ] Blank line: There should be two blank lines between top-level definitions, such as function or class definitions. There should be a blank line between the method definition, the class definition and the first method. In the function or method, if you think there is a logical break, leave a blank line;

- [ ] Line length: Each line of code should not exceed 80 characters, which means that code can be seen completely after splitting into two screens. Python will implicitly join the lines in [parentheses, square brackets and curly braces](http://docs.python.org/2/reference/lexical_analysis.html#implicit-line-joining). You can make use of this feature by adding a pair of extra parentheses around the expression instead of using backslashes.

- [ ] Brackets: Brackets can be used for line connection, but do not use unnecessary brackets in statement;

- [ ] Branch: Each statement must be on its own line, do not use semicolons.

- [ ] Naming: Module name: `module_name`; Package name: `package_name`; Class name: `ClassName`; Method name: `method_name`; Exception name: `ExceptionName`; Function name: `function_name`; Global Constant name: `GLOBAL_CONSTANT_NAME`; Global variable name: `global_var_name`; Instance name: `instance_var_name`; Function parameter name: `function_parameter_name`; Local variable name: `local_var_name`


## 2 Expandable module standard

Currently PaddleSeg supports the component expansion including `model, loss, backbone, transform, dataset`. Among them, the specifications of backbone and model are similar, the transform specifivation is relatively simple. Therefore, the following description mainly specify the standard of model, loss, and dataset expansion.

### 2.1 Model

This part we use PSPNet as an example to illustrate. To develop `PSPNet`, you need to create ```pspnet.py``` in the ```paddleseg/models``` directory. please notice that the file names are all lowercase. And the content of the entire file is divided into three parts, the `copyright` part, the `import` part, and the model implementation part. The first two parts are illustrated in the above content.

Model implementation normally includes three parts: the main model, the segmentation head, and the auxiliary module. If the model does not have a backbone, there are only main model and auxiliary modules.

#### 2.1.1. Main model

##### 2.1.1.1 Model declaration specification

This part is the first part of the model after you import models.

1. Use manager to add the main model, that is, add the following statement before the main model definition:

   Note: **Only** the main model requires the manager decorator.

```python
@manager.MODELS.add_component
class PSPNet(nn.Layer):
```

2. Inherit nn.Layer;

3. Add english comment:

   1. Add "`The xxx implementation based on PaddlePaddle.`";
   2. Add "`The original article refers to`" + author name and article name + article link;
   3. Specify the parameter type. If it is optional, then add `optional` keyward, and then add "`Default: xx`" at the end of the parameter comment.
   4. If possible, you can further add `Returns, Raises` to explain the return value of the function/method and possible errors.

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
        backbone_indices (tuple, optional): Two values ​​in the tuple indicate the indices of output of backbone.
        pp_out_channels (int, optional): The output channels after Pyramid Pooling Module. Default: 1024.
        bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1,2,3,6).
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """
```

##### 2.1.1.2 __init__ specification

1. Add parameter list, arrage them in the following order: `num_classes, backbone, backbone_indices, ......, align_corners, in_channels, pretrained`. The order of other other intermediate parameters can be adjusted freely;
2. Parameter names should have meaning: Try to avoid names with no obvious meaning, such as n, m, aa, unless it follows the original implementation;
3. All the parameters in ```__init__``` should be written out explicitly, you should not include variable length parameters such as: `*args, **kwargs`;
4. ```super().__init__()``` should not have parameters;
5. At the end, call ```self.init_weight()``` to load the pretrained weight by `pretrained` params;
6. If the model does not have backbone, it must have the input params of `in_channels`, which denotes the channels of input image. The `in_channels` is set as 3 in default.
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

##### 2.1.1.3 Forward specification

1. The logic should be as concise as possible and make as many as component calls.
2. Resize the output to the original image size and return them in the form of a list. The first element of the list is the main output, and the others are auxiliary outputs.
3. If the execution branch is different during model training and prediction, use self.training variable in if statement to implement different branches (for example: bisnetv2 model).
4. To obtain the shape of the Tensor, it is recommended to use `paddle.shape(x)` instead of `x.shape` to avoid errors in exporting inference model.
5. The network forward pass dose not support tensor->numpy->tensor operation.

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

##### 2.1.1.4 init_weight specification

1. Call `load_entire_model` to load pretrained model for model weights initialization.
2. Models without backbone can be initialized by calling `param_init` in `paddleseg.cvlib`.

    ```python
    # Load the entire model with backbone
    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    # Initialization model without backbone
    def init_weight(self):
        """Initialize the parameters of model parts."""
        for sublayer in self.sublayers():
            if isinstance(sublayer, nn.Conv2D):
            param_init.normal_init(sublayer.weight, std=0.001)
            elif isinstance(sublayer, (nn.BatchNorm, nn.SyncBatchNorm)):
            param_init.constant_init(sublayer.weight, value=1.0)
            param_init.constant_init(sublayer.bias, value=0.0)
    ```
#### 2.1.2 backbone

The implementation of backbone is the same as the model, and please refer to `paddleseg/models/backbones/mobilenetv2.py` for more details.

The `__init__` function of backbone must have the params of `in_channels=3`, which denotes the channels of input image.

Generally, the backbone has several output feature maps, of which the size are 1/4, 1/8, 1/16 and 1/32 of the input image.

The backbone class must has `self.feat_channels` attribute, and it denotes the channels of output feature maps.

The backbone has different size, so we employ several function registerd by `@manager.BACKBONES.add_component` to define them as following.

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

#### 2.1.3 segmentation head

At present, the model in PaddleSeg only has a single segmentation head, so the segmentation head module is named as model name + Head. And the annotation specification is consistent with the main model.

```python
class PSPNetHead(nn.Layer):
```

If your model is a lightweight one without backbone, it can be treated as model with single segmentation head. For simplicity, you can write the code in the main model instead of the head.

#### 2.1.4 auxiliary module

Other segments except for the main model and the segmentation header are called as auxiliary modules. Currently, PaddleSeg has provided common auxiliary modules, such as `SyncBN, ConvBNReLU, FCN (AuxLayer), PPModule, ASPP, AttentionBlock` and etc. You can refer to  `paddleseg/models/layers` for details.
1. **Must** use the built-in auxiliary module of PaddleSeg if you can;
2. If the built-in auxiliary modules do not meet the requirements, you can customize a module;
3. The custom module should comply with a good code style and have comments on key modules;
4. Please comment the name of the custom module in your PR.


After complete the model file, add import information in ```paddleseg/models/__init__.py```. If there is no addition of other loss, the process of model development is completed.

```python
from .pspnet import *
```

### 2.2 Loss development specification

The loss specification takes `paddleseg/models/losses/cross_entropy_loss.py` as an example:

Loss declaration specification:
1. Use the `manager` decorator on the loss head;
2. Inherit `nn.Layer`;
3. Add English notes:
   1. Loss meaning: what does the class do, what is the loss expression, and what is the improvement compare to other losses (optional)
   2. Loss parameters: The loss parameters are flexible, you can specify the weight, `ignore_index`, and etc.
4. Must support setting `ignore_index` to ignore the special vale in label.
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
            top_k_percent_pixels (float, optional): the value lies in [0.0, 1.0]. When its value <1.0, only compute the loss for
                the top k percent pixels (e.g., the top 20% pixels). This is useful for hard pixel mining. Default ``1.0``.
            data_format (str, optional): The tensor format to use,'NCHW' or'NHWC'. Default ``'NCHW'``.
        """
    ```



### 2.3 Dataset development specification

This part we takes `paddleseg/dataset/cityscapes.py` as an example. In thi dataset file, only one class with the same name as the dataset is declared. And you should create `datasetname.py` in `paddleseg/dataset`.

#### 2.3.1 Dataset declaration specification

1. Add a decorator to the head of the class;

   ```python
   @manager.DATASETS.add_component
   ```

2. The class inherits the `Dataset` base class;

3. The document part describes the source of the dataset, the structure of the data set, and the meaning of the parameters.

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
            | |--train
            | |--val
            | |--test
            |
            |--gtFine
            | |--train
            | |--val
            | |--test
        Make sure there are **labelTrainIds.png in gtFine directory. If not, please run the conver_cityscapes.py in tools.
        Args:
            transforms (list): Transforms for image.
            dataset_root (str): Cityscapes dataset directory.
            mode (str, optional): Which part of dataset to use. it is one of ('train','val','test'). Default:'train'.
            edge (bool, optional): Whether to compute edge while training. Default: False
        """
    ```

#### 2.3.2 __init__ specification

1. All the parameters in ```__init__``` are written out explicitly, you **cannot** include variable length parameters such as: ```*args, **kwargs```;
2. ```super().__init__()``` should not have parameters;
3. The order of the parameters is consistent with the above example;
4. By creating ```self.file_list``` in the ```__init__``` method, the dataset can read images according to the path in it.

## 3 Export and test the inference model

To develop a model, we need to not only pay attention to the accuracy of the model, but also check the correctness of the exported model to accelerate the inference speed. Only when the model can be successfully deployed can a model count as truly developed.

### 3.1 Export the prediction model

Model develop based on PaddlePaddle's dynamic graph. We need to export the dynamic graph model to a static graph for prediction.

The dynamic graph model can be exported using dynamic-to-static technology, which will not be introduced here. For specific instructions, please refer to [document](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/04_dygraph_to_static/index_cn.html).

Please refer to [document](../../model_export.md) to export the static prediction model. If no error is reported, the prediction model will be saved to the specified directory. If an error is reported, modify the network code according to the log and export again.

### 3.2 Test the prediction model

Please refer to [document](../../deployment/inference/python_inference.md) to test model. Use Paddle Inference  API on X86 CPU or NV GPU to load the prediction model, load the image for testing, and check whether the segmentation result image is correct.


## 4 PR checklist
* Follow the code submission process according to [Code Submission Specification](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/docs/pr/pr/pr.md), including pulling the latest content and switching branches.
* Create a subdirectory (```pspnet```) named after the model name in the ```configs``` directory. The subdirectory contsists of yml configuration files and readme.md, please refer to the [demo](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/configs/pspnet)
* The name of yml configuration file should be `model name + backbone + out_stride + data set + training resolution + training iters.yml`, and the parts that is not included should be ignored. For details, please refer to [Configuration Item Document](../../design/use/use_cn.md).
* In readme.md, the reference style of the model should adopts Chicago, that is, the names of all authors. For example:```Zhao, Hengshuang, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia. "Pyramid scene parsing network." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2881-2890. 2017.`` `
* In readme.md, provide the training and test performance on at least one dataset in the following format.
    * `Resolution` denotes the crop_size in training dataset
    * `mIoU, mIoU(flip), mIoU(ms+flip)` are the results of evaluating the model. `ms` means `multi-scale`, that is, three kinds of `scale` [0.75, 1.0, 1.25] are used; `flip` means horizontal flip. For detailed evaluation, please refer to [Model Evaluation](../../evaluation/evaluate_cn.md)
    * Provide download links including: trained model parameters, training log, training vdl.

    | Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    ||||||||[model]() \| [log]() \| [vdl]()|

* Refer to the `New file self-inspection` and `Expandable module standard` in the above to check and refactor all new files and expandable modules.
* Finish the test of `Export and test the inference model` in the above and provide the results in PR for reviewers.

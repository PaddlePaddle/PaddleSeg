# Model Distillation Tutorial

# 1. Introduction

Based on the idea of teacher and student, model distillation uses big teacher model to teach small student model in the training stage, which is a common method of model compression.
Compared to training the small model alone, model distillation is usually benficial for higher accuracy.
If you are interested in the theory of model distillation, there are a survey in [arxiv](https://arxiv.org/abs/2006.05525).

Based on PaddleSlim, PaddleSeg provides the module of model distillation. The key points of using model distillation are as follows:
* Chose the teacher and student models
* Train the teacher model
* Set the config files of model distillation
* Training of model distillation, namely train the student with the guidance of the teacher model

In this tutorial, we demonstrate a demo of model distillation, and then present the advanced usage of model distillation.

# 2. Model Distillation Demo

### 2.1 Requirements

Please follow [installation document](../../install.md) to install the requirements of PaddleSeg.

Besides, run the following instructions to install PaddleSlim.

```shell
git clone https://github.com/PaddlePaddle/PaddleSlim.git

# checkout to special commit
git reset --hard 15ef0c7dcee5a622787b7445f21ad9d1dea0a933

# install
python setup.py install
```

### 2.2 Chose the Teacher and Student Models

In this demo, DeepLabV3P_ResNet50_vd is the teacher model and  DeepLabV3P_ResNet18_vd is the student model. Besides, we use the optic disc segmentation dataset for simplicity.

### 2.3 Train the Teacher Model

The config file of the teacher model is `PaddleSeg/configs/quick_start/deeplabv3p_resnet50_os8_optic_disc_512x512_1k_teacher.yml`.

Run the following instructions in the root directory of PaddleSeg to train the teacher model.

```shell
export CUDA_VISIBLE_DEVICES=0  # Set GPU for Linux
# set CUDA_VISIBLE_DEVICES=0   # Seg GPU for Windows

python train.py \
    --config configs/quick_start/deeplabv3p_resnet50_os8_optic_disc_512x512_1k_teacher.yml \
    --do_eval \
    --use_vdl \
    --save_interval 250 \
    --num_workers 3 \
    --seed 0 \
    --save_dir output/deeplabv3p_resnet50
```

After the traing, the mIoU of the teacher model is 91.54% and the trained weights are saved in `output/deeplabv3p_resnet50/best_model/model.pdparams`.

### 2.4 Train the Student Model (Optional)

In this step, we train the student model without the guidance of the teacher model.  

The config file of the student model is `PaddleSeg/configs/quick_start/deeplabv3p_resnet18_os8_optic_disc_512x512_1k_student.yml`.

Run the following instructions in the root directory of PaddleSeg to train the student model alone.

```shell
export CUDA_VISIBLE_DEVICES=0  # Set GPU for Linux
# set CUDA_VISIBLE_DEVICES=0   # Seg GPU for Windows

python train.py \
    --config configs/quick_start/deeplabv3p_resnet18_os8_optic_disc_512x512_1k_student.yml \
    --do_eval \
    --use_vdl \
    --save_interval 250 \
    --num_workers 3 \
    --seed 0 \
    --save_dir output/deeplabv3p_resnet18
```

The mIoU of the student model is 83.93% and the trained weights are saved in `output/deeplabv3p_resnet18/best_model/model.pdparams`.


### 2.5 Set the Config File of Model Distillation

The training of model distillation needs the config files of the teacher and student models.

We open the teacher config file (`PaddleSeg/configs/quick_start/deeplabv3p_resnet50_os8_optic_disc_512x512_1k_teacher.yml`) and set the pretrained in the last line as the path of the teacher model's weights as follows.

```shell
model:
  type: DeepLabV3P
  backbone:
    type: ResNet50_vd
    output_stride: 8
    multi_grid: [1, 2, 4]
    pretrained: Null
  num_classes: 2
  backbone_indices: [0, 3]
  aspp_ratios: [1, 12, 24, 36]
  aspp_out_channels: 256
  align_corners: False
  pretrained: output/deeplabv3p_resnet50/best_model/model.pdparams
```

It is not necessary to modify the config file of the student model. Note that, the config file has normal loss and distillation loss.

```shell
loss:
  types:
    - type: CrossEntropyLoss
  coef: [1]

# distill_loss is used for distillation
distill_loss:
  types:
    - type: KLLoss
  coef: [3]
```

### 2.6 Training of Model Distillation

With the config files of the teacher and student models, run the following instructions in the root directory of PaddleSeg to train the student model with the guidance of the teacher model.

```shell
export CUDA_VISIBLE_DEVICES=0  # Set GPU for Linux
# set CUDA_VISIBLE_DEVICES=0   # Seg GPU for Windows

python slim/distill/distill_train.py \
       --teather_config ./configs/quick_start/deeplabv3p_resnet50_os8_optic_disc_512x512_1k_teacher.yml \
       --student_config ./configs/quick_start/deeplabv3p_resnet18_os8_optic_disc_512x512_1k_student.yml \
       --do_eval \
       --use_vdl \
       --save_interval 250 \
       --num_workers 3 \
       --seed 0 \
       --save_dir output/deeplabv3p_resnet18_distill
```

The script of `slim/distill/distill_train.py` creates the teacher model, creates the student model, loads dataset to train the student model while the teacher model is fixed.

After the training, the mIoU of the student model is 85.79% and the trained weights are saved in `output/deeplabv3p_resnet18_distill/best_model`.

Compared the accuracy of these two student models, the model distillation imporves the mIoU by 1.86%.

## 3. Advanced Usage of Model Distillation

### 3.1 Single-Machine Multiple-GPUs Training

In order to accelerate the training of model distillation with single machine multiple GPUs, we export `CUDA_VISIBLE_DEVICES` and use `paddle.distributed.launch` to start the script as follows. Note that, PaddlePaddle does not support single machine multiple GPUs training on Windows.

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3 # use four GPUs

python -m paddle.distributed.launch slim/distill/distill_train.py \
       --teather_config ./configs/quick_start/deeplabv3p_resnet50_os8_optic_disc_512x512_1k_teacher.yml \
       --student_config ./configs/quick_start/deeplabv3p_resnet18_os8_optic_disc_512x512_1k_student.yml \
       --do_eval \
       --use_vdl \
       --save_interval 250 \
       --num_workers 3 \
       --seed 0 \
       --save_dir output/deeplabv3p_resnet18_distill
```

### 3.2 The Weights of Losses

In the config file of the student model, the `coef` means the weight of the according loss, such as the normal loss and distill_loss.
You can adjust the weights of different losses to imporve the accuracy.


### 3.3 Use Intermediate Tensors for Distillation

Model distillation only utilizes the output tensors of the teacher and student models in the above demo for simplicity.
In fact, we can also use intermediate tensors for model distillation.

* Chose the intermediate tensors in the teacher and student models

It requires the intermediate tensors in the teacher and student models have the same shape for now.

* Set the intermediate tensors for distillation

In Paddeseg, the `slim/distill/distill_config.py` file has a "prepare_distill_adaptor" function. We utilize the StudentAdaptor and TeatherAdaptor class to set the intermediate tensors for model distillation.

Generally speaking, PaddlePaddle has two types of api. The first type is layer api, of which the base class is "paddle.nn.Layer", such as "paddle.nn.Conv2D". The second type is function api, such as paddle.reshape.

If the intermediate tensor is the output of layer api, we set the `mapping_layers['name_index'] = 'layer_name'` outside the block of `if self.add_tensor`.
If the intermediate tensor is the output of function api, we set the `mapping_layers['name_index'] = 'tensor_name'.` inside the block of `if self.add_tensor`.


```Python
def prepare_distill_adaptor():
    """
    Prepare the distill adaptors for student and teacher model.
    The adaptors set the intermediate feature tensors that used for distillation.
    """

    class StudentAdaptor(AdaptorBase):
        def mapping_layers(self):
            mapping_layers = {}
            # the interior tensor is the output of layer api
            # mapping_layers['hidden_0'] = 'layer_name'
            if self.add_tensor:
                # the interior tensor is the output of function api
                # mapping_layers["hidden_0"] = self.model.logit_list
                pass
            return mapping_layers

    class TeatherAdaptor(AdaptorBase):
        def mapping_layers(self):
            mapping_layers = {}
            # mapping_layers['hidden_0'] = 'layer_name'
            if self.add_tensor:
                # mapping_layers["hidden_0"] = self.model.logit_list
                pass
            return mapping_layers

    return StudentAdaptor, TeatherAdaptor
```

For example, The output tensors of the "nn.Conv2D" (layer api) and the "paddle.reshape" (function api) are unsed for distillation in the next model.
Then, the corresponding StudentAdaptor is showed as follows.

```python
class Model(nn.Layer):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2D(3, 3, 3, padding=1)
        self.fc = nn.Linear(3072, 10)

    def forward(self, x):
        conv1_out = self.conv1(x)
        self.reshape_out = paddle.reshape(conv1_out, shape=[x.shape[0], -1])  # note that `self.reshape_out`
        out = self.fc(self.reshape_out)
        return out
```

```python
class StudentAdaptor(AdaptorBase):
    def mapping_layers(self):
        mapping_layers = {}
        mapping_layers['hidden_0'] = 'conv1'   # The output of layer api
        if self.add_tensor:
            mapping_layers["hidden_1"] = self.model.reshape_out # The output of function api
        return mapping_layers
```

* Set the config of Distillation

Follow the above example, we define the "prepare_distill_config" function in `slim/distill/distill_config.py` to set the config of distillation.
In detail, the feature_type and s_feature_idx determine the tensor name in student model. The feature_type and t_feature_idx determine the tensor name in teacher model. The loss_function determine the type of distillation loss.

```python
def prepare_distill_config():
    """
    Prepare the distill config.
    """
    config_1 = {
        'feature_type': 'hidden',
        's_feature_idx': 0,
        't_feature_idx': 0,
        'loss_function': 'SegChannelwiseLoss',
        'weight': 1.0
    }
    config_2 = {
        'feature_type': 'hidden',
        's_feature_idx': 1,
        't_feature_idx': 1,
        'loss_function': 'SegChannelwiseLoss',
        'weight': 1.0
    }
    distill_config = [config_1, config_2]

    return distill_config
```

* Training for Distillation

Use the same method as above to run the `slim/distill/distill_train.py`.

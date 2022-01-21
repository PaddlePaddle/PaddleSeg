English | [简体中文](use_cn.md)
# Configuration item

----
### train_dataset
* Training dataset
>
>  * Args
>     * type : Dataset type, please refer to [Data Set Document](../../apis/datasets/datasets.md) for the supported values
>     * **others** : Please refer to [Training Configuration File of Corresponding Model](../../../configs)

----
### val_dataset
* Validation dataset
>  * Args
>     * type : Dataset type, please refer to [Data Set Document](../../apis/datasets/datasets.md) for the supported values
>     * **others** : Please refer to [Training Configuration File of Corresponding Model](../../../configs)
>

----
### batch_size
* On a single card, the amount of data during each iteration of training. Generally speaking, the larger the video memory of the machine you are using, the larger the batch_size value.

----
### iters
* The process of using a batch of data to update the parameters of the semantic segmentation model is called one training, that is, one iteration. Iters is the number of iterations in the training process.

----
### optimizer
* Optimizer in training
>  * Args
>     * type : Optimizer type, currently only supports'sgd' and'adam'
>     * momentum : Momentum optimization.
>     * weight_decay : L2 regularized value.

----
### lr_scheduler
* Learning rate
>  * Args
>     * type : Learning rate type, supports 12 strategies: 'PolynomialDecay', 'PiecewiseDecay', 'StepDecay', 'CosineAnnealingDecay', 'ExponentialDecay', 'InverseTimeDecay', 'LinearWarmup', 'MultiStepDecay', 'NaturalExpDecay', 'NoamDecay', ReduceOnPlateau, LambdaDecay.
>     * **others** : Please refer to [Paddle official LRScheduler document](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/LRScheduler_cn.html)

----
### learning_rate（This configuration is not recommended and will be obsolete in the future. It is recommended to use `lr_scheduler` instead）
* Learning rate
>  * Args
>     * value : Initial learning rate.
>     * decay : Attenuation configuration.
>       * type : Attenuation type, currently only supports poly.
>       * power : Attenuation rate.
>       * end_lr : Final learning rate.

----
### loss
* Loss function
>  * Args
>     * types : List of loss functions.
>       * type : Loss function type, please refer to the loss function library for the supported values.
>     * coef : List of coefficients corresponding to the loss function list.

----
### model
* Model to be trained
>  * Args
>     * type : Model type, please refer to [Model Library](../../apis/models/models.md) for the supported values
>     * **others** : Please refer to [Training Configuration File of Corresponding Model](../../../configs)
---
### export
* Model export configuration
>  * Args
>    * transforms : The preprocessing operation during prediction, the supported transforms are the same as `train_dataset`, `val_dataset`, etc. If you do not fill in this item, only the data will be normalized by default.

# Example

```yaml
batch_size: 4 # Set the number of pictures sent to the network at one iteration. Generally speaking, the larger the video memory of the machine you are using, the higher the batch_size value.
iters: 80000 # Number of iterations

train_dataset: # Training dataset
  type: Cityscapes # The name of the training dataset class
  dataset_root: data/cityscapes # The directory where the training dataset is stored
  transforms: # Data transformation and data augmentation
    - type: ResizeStepScaling # The image is scaled according to a certain ratio, and this ratio takes scale_step_size as the step size
      min_scale_factor: 0.5 # Parameters involved in the scaling process
      max_scale_factor: 2.0
      scale_step_size: 0.25
    - type: RandomPaddingCrop # Random cropping of images and annotations
      crop_size: [1024, 512]
    - type: RandomHorizontalFlip # Flip the image horizontally with a certain probability
    - type: Normalize # Normalize the image
  mode: train # Training mode

val_dataset: # Validation dataset
  type: Cityscapes # The name of the validating dataset class
  dataset_root: data/cityscapes # The directory where the validating dataset is stored
  transforms:
    - type: Normalize # Normalize the image
  mode: val # Validating mode

optimizer: # Which optimizer to use
  type: sgd # Stochastic gradient descent
  momentum: 0.9
  weight_decay: 4.0e-5

lr_scheduler: # Related settings for learning rate
  type: PolynomialDecay # A type of learning rate,a total of 12 strategies are supported
  learning_rate: 0.01
  power: 0.9
  end_lr: 0

loss: # What loss function to use
  types:
    - type: CrossEntropyLoss # Cross entropy loss function
  coef: [1] # When multiple loss functions are used, the ratio of each loss can be specified in coef

model: # Which semantic segmentation model to use
  type: FCN
  backbone: # What kind of backbone network to use
    type: HRNet_W18
    pretrained: pretrained_model/hrnet_w18_ssld # Specify the storage path of the pre-trained model
  num_classes: 19 # Number of pixel categories
  pretrained: Null
  backbone_indices: [-1]

```

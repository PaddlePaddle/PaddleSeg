English | [简体中文](README_cn.md)

The config files of different models are saved in `PaddleSeg/configs`.
PaddleSeg use the config files to train, validate and export models.
# Configuration items

----
### train_dataset
>  Training datasset
>
>  * parameter
>     * type: Dataset type, please refer to the training configuration file for more details of supported values
>     * **others**: Please refer to the corresponding model training configuration file

----
### val_dataset
>  Evaluation dataset
>  * parameter
>     * type: Dataset type, please refer to the training configuration file for more details of supported values
>     * **others**: Please refer to the corresponding model training configuration file
>

----
### batch_size
>  On a single card, the amount of data during each iteration of training

----
### iters
>  Training steps

----
### optimizer
> Training optimizer
>  * parameter
>     * type : supports all official optimizers of PaddlePaddle
>     * weight_decay : L2 regularization value
>     * **others** : Please refer to [Optimizer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Overview_cn.html)

----
### lr_scheduler
> Learning rate
>  * parameter
>     * type : learning rate type, supports 10 strategies, namely 'PolynomialDecay', 'PiecewiseDecay', 'StepDecay', 'CosineAnnealingDecay', 'ExponentialDecay', 'InverseTimeDecay', 'LinearWarmup', 'MultiStepDecay', 'NaturalExpDecay', 'NoamDecay'.
>     * **others** : Please refer to [Paddle official LRScheduler document](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/LRScheduler_cn.html)

----
### learning_rate（this configuration is not recommended, it will be discarded in the future, we recommend to use lr_scheduler instead）
> Learning rate
>  * parameter
>     * value: initial learning rate value
>     * decay: decay configuration
>       * type: attenuation type, currently only supports poly
>       * power: attenuation rate
>       * end_lr: final learning rate

----
### loss
> Loss function
>  * parameter
>     * types: list of loss functions
>       * type:  Loss function type, please refer to the loss function library for more details
>       * ignore_index : The category that needs to be ignored during the training process. The default value is the same train_datasetas ignore_index. It is recommended not to set this item . If you set this, "ignore_index" in loss and train_datasetthe must be the same.
>     * coef : a list of coefficients corresponding to corresponding loss functions

----
### model
> Model to be trained
>  * parameter
>     * type : model type, please refer to the model library for the more details
>     * **others**: Please refer to the corresponding model training configuration file
---
### export
> Model export configuration
>  * parameter
>    * transforms: Preprocessing operations during prediction. The transforms are the same as train_dataset, val_datasetetc. If you do not fill in this item, the data will be normalized by default.

For more details, please refer to [detailed configuration file](../docs/design/use/use.md)

# paddleseg.core

The interface for training, evaluation and prediction.
- [Training](#Training)
- [Evaluation](#Evaluation)
- [Prediction](#Prediction)

## [Training](../../paddleseg/core/train.py)
> paddleseg.core.train(model, train_dataset, val_dataset=None, optimizer=None, save_dir='output', iters=10000, batch_size=2, resume_model=None, save_interval=1000, log_iters=10, num_workers=0, use_vdl=False, losses=None)

    Launch training.

> Args
> > - **mode**l（nn.Layer): A sementic segmentation model.
> > - **train_dataset** (paddle.io.Dataset): Used to read and process training datasets.
> > - **val_dataset** (paddle.io.Dataset, optional): Used to read and process validation datasets.
> > - **optimizer** (paddle.optimizer.Optimizer): The optimizer.
> > - **save_dir** (str, optional): The directory for saving the model snapshot. Default: 'output'.
> > - **iters** (int, optional): How may iters to train the model. Defualt: 10000.
> > - **batch_size** (int, optional): Mini batch size of one gpu or cpu. Default: 2.
> > - **resume_model** (str, optional): The path of resume model.
> > - **save_interval** (int, optional): How many iters to save a model snapshot once during training. Default: 1000.
> > - **log_iters** (int, optional): Display logging information at every log_iters. Default: 10.
> > - **num_workers** (int, optional): Num workers for data loader. Default: 0.
> > - **use_vdl** (bool, optional): Whether to record the data to VisualDL during training. Default: False.
> > - **losses** (dict): A dict including 'types' and 'coef'. The length of coef should equal to 1 or len(losses['types']).
    The 'types' item is a list of object of paddleseg.models.losses while the 'coef' item is a list of the relevant coefficient.

## [Evaluation](../../paddleseg/core/val.py)
> paddleseg.core.evaluate(model, eval_dataset, aug_eval=False, scales=1.0, flip_horizontal=True, flip_vertical=False, is_slide=False, stride=None, crop_size=None, num_workers=0)

    Launch evaluation.

> Args
> > - **model**（nn.Layer): A sementic segmentation model.
> > - **eval_dataset** (paddle.io.Dataset): Used to read and process validation datasets.
> > - **aug_eval** (bool, optional): Whether to use mulit-scales and flip augment for evaluation. Default: False.
> > - **scales** (list|float, optional): Scales for augment. It is valid when `aug_eval` is True. Default: 1.0.
> > - **flip_horizontal** (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_eval` is True. Default: True.
> > - **flip_vertical** (bool, optional): Whether to use flip vertically augment. It is valid when `aug_eval` is True. Default: False.
> > - **is_slide** (bool, optional): Whether to evaluate by sliding window. Default: False.
> > - **stride** (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
        It should be provided when `is_slide` is True.
> > - **crop_size** (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
        It should be provided when `is_slide` is True.
> > - **num_workers** (int, optional): Num workers for data loader. Default: 0.

> Returns
> > - **float**: The mIoU of validation datasets.
> > - **float**: The accuracy of validation datasets.

## [Prediction](../../paddleseg/core/predict.py)

> paddleseg.core.predict(model, model_path, transforms, image_list, image_dir=None, save_dir='output', aug_pred=False, scales=1.0, flip_horizontal=True, flip_vertical=False, is_slide=False, stride=None, crop_size=None)

    Launch predict and visualize.

> Args
> > - **model** (nn.Layer): Used to predict for input image.
> > - **model_path** (str): The path of pretrained model.
> > - **transforms** (transform.Compose): Preprocess for input image.
> > - **image_list** (list): A list of image path to be predicted.
> > - **image_dir** (str, optional): The root directory of the images predicted. Default: None.
> > - **save_dir**** (bool, optional): Whether to use mulit-scales and flip augment for predition. Default: False.
> > - **scales** (list|float, optional): Scales for augment. It is valid when `aug_pred` is True. Default: 1.0.
> > - **flip_horizontal** (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_pred` is True. Default: True.
> > - **flip_vertical** (bool, optional): Whether to use flip vertically augment. It is valid when `aug_pred` is True. Default: False.
> > - **is_slide** (bool, optional): Whether to predict by sliding window. Default: False.
> > - **stride** (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
    It should be provided when `is_slide` is True.
> > - **crop_size** (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
    It should be provided when `is_slide` is True.

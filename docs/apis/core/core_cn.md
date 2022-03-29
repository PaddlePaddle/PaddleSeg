简体中文 | [English](core.md)
# paddleseg.core

启动模型训练，评估与预测的接口
- [train](#train)
- [evaluate](#evaluate)
- [predict](#predict)

## [train](../../../paddleseg/core/train.py)
```python
paddleseg.core.train(
        model,
        train_dataset,
        val_dataset = None,
        optimizer = None,
        save_dir = 'output',
        iters = 10000,
        batch_size = 2,
        resume_model = None,
        save_interval = 1000,
        log_iters = 10,
        num_workers = 0,
        use_vdl = False,
        losses = None
)
```
> 启动模型的训练。

### 参数
* **model**（nn.Layer): 指定一个语义分割模型。
* **train_dataset** (paddle.io.Dataset): 用于读取和处理训练数据集。
* **val_dataset** (paddle.io.Dataset, optional): 用于读取和处理验证数据集。
* **optimizer** (paddle.optimizer.Optimizer): 指定优化器。
* **save_dir** (str, optional): 保存模型快照的目录位置。*默认: 'output'*
* **iters** (int, optional): 训练模型时的迭代轮数。 *默认: 10000*
* **batch_size** (int, optional): 一个 gpu 或 cpu 的最小批量大小。 *默认: 2*
* **resume_model** (str, optional): 加载恢复模型的路径。
* **save_interval** (int, optional): 指定在训练中多少轮保存一次模型快照。*默认: 1000*
* **log_iters** (int, optional): 每经过 log_iters 轮迭代显示一次日志信息。 *默认: 10*
* **num_workers** (int, optional): 指定num workers的数目，即进程的数目。用以多进程读取图片。*默认: 0*
* **use_vdl** (bool, optional): 在训练过程中是否将数据记录在VisualDL里。 *默认: False*
* **losses** (dict): 一个包含'types'和'coef'的字典。 coef的长度或为1，或等同于len(losses['types'])。
    'types' 项是 paddleseg.models.losses 对象构成的列表，而 'coef' 项是相关的系数的列表。

## [evaluate](../../../paddleseg/core/val.py)
```python
paddleseg.core.evaluate(
        model,
        eval_dataset,
        aug_eval = False,
        scales = 1.0,
        flip_horizontal = True,
        flip_vertical = False,
        is_slide = False,
        stride = None,
        crop_size = None,
        num_workers = 0
)
```
> 启动模型的评估。

### 参数
* **model**（nn.Layer): 指定一个语义分割模型。
* **eval_dataset** (paddle.io.Dataset): 用于读取和处理评估数据集。
* **aug_eval** (bool, optional): 是否使用多尺度和翻转增强进行评估。 *默认: False*
* **scales** (list|float, optional): 数据增强的尺度。 当 `aug_eval` 为 True 时有效。 *默认: 1.0*
* **flip_horizontal** (bool, optional): 是否使用水平翻转增强。 当 `aug_eval` 为 True 时有效。 *默认: True*
* **flip_vertical** (bool, optional): 是否使用垂直翻转增强。 当`aug_eval` 为 True 时有效。 *默认: False*
* **is_slide** (bool, optional): 是否通过滑动窗口进行评估。 *默认: False*
* **stride** (tuple|list, optional): 滑动窗口的步长，第一维参数指定宽度，第二维参数指定高度。
        当`is_slide` 为True时，应该指定该参数。
* **crop_size** (tuple|list, optional):  滑动窗口的裁剪大小，第一维参数指定宽度，第二维参数指定高度。
        当`is_slide` 为True时，应该指定该参数.
* **num_workers** (int, optional): 指定num workers的数目，即进程的数目。用以多进程读取图片。*默认: 0*

### 返回值
* **float**: 验证数据集的mIoU。
* **float**: 验证数据集的accuracy。
* **float**: 验证数据集的kappa系数。

## [predict](../../../paddleseg/core/predict.py)
```python
paddleseg.core.predict(
        model,
        model_path,
        transforms,
        image_list,
        image_dir = None,
        save_dir = 'output',
        aug_pred = False,
        scales = 1.0,
        flip_horizontal = True,
        flip_vertical = False,
        is_slide = False,
        stride = None,
        crop_size = None
)
```
> 启动预测与可视化。

### 参数
* **model** (nn.Layer): 用以对输入图像进行预测。
* **model_path** (str): 预训练模型的路径。
* **transforms** (transform.Compose): 对输入图像的预处理。
* **image_list** (list): 预测数据集的存放路径列表。
* **image_dir** (str, optional): 预测数据集的根目录。 *默认: None*
* **save_dir**** (bool, optional): 是否使用多尺度与翻转增强进行预测。*默认: False*
* **scales** (list|float, optional): 数据增强的尺度。 当 `aug_pred` 为 True 时有效。 *默认: 1.0*
* **flip_horizontal** (bool, optional): 是否使用水平翻转增强。 当 `aug_pred` 为 True 时有效。 *默认: True*
* **flip_vertical** (bool, optional): 是否使用垂直翻转增强。 当`aug_pred` 为 True 时有效。 *默认: False*
* **is_slide** (bool, optional): 是否通过滑动窗口进行评估。 *默认: False*
* **stride** (tuple|list, optional): 滑动窗口的步长，第一维参数指定宽度，第二维参数指定高度。
        当`is_slide` 为True时，应该指定该参数。
* **crop_size** (tuple|list, optional):  滑动窗口的裁剪大小，第一维参数指定宽度，第二维参数指定高度。
        当`is_slide` 为True时，应该指定该参数。
* **num_workers** (int, optional): 指定num workers的数目，即进程的数目。用以多进程读取图片。*默认: 0*

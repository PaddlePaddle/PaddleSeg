English|[简体中文](evaluate_cn.md)
## Model Evaluating

### 1. Evaluation and Prediction under **Configuration-Driven** Approach

#### Evaluating

After the training, the user can use the evaluation script val.py to evaluate the effect of the model. Assuming that the number of iterations (iters) in the training process is 1000, the interval for saving the model is 500, that is, the training model is saved twice for every 1000 iterations of the data set. Therefore, a total of 2 regularly saved models will be generated, plus the best saved model `best_model`, there are a total of 3 models, and the model file that you want to evaluate can be specified by `model_path`.

```
!python val.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --model_path output/iter_1000/model.pdparams
```

If you want to perform multi-scale flip evaluation, you can turn it on by passing in `--aug_eval`, and then passing in scale information via `--scales`, `--flip_horizontal` turns on horizontal flip, and `flip_vertical` turns on vertical flip. Examples are as follows:

```
python val.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --model_path output/iter_1000/model.pdparams \
       --aug_eval \
       --scales 0.75 1.0 1.25 \
       --flip_horizontal
```

If you want to perform sliding window evaluation, you can open it by passing in `--is_slide`, pass in the window size by `--crop_size`, and pass in the step size by `--stride`. Examples of usage are as follows:

```
python val.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --model_path output/iter_1000/model.pdparams \
       --is_slide \
       --crop_size 256 256 \
       --stride 128 128
```

In the image segmentation, evaluating model quality is mainly judged by three indicators, `accuracy` (acc), `mean intersection over union` (mIoU), and `Kappa coefficient`.

- **Accuracy**: refers to the proportion of pixels with correct category prediction to the total pixels. The higher the accuracy, the better the quality of the model.
- **Average intersection ratio**: perform inference calculations for each category dataset separately, divide the calculated intersection of the predicted area and the actual area by the union of the predicted area and the actual area, and then average the results of all categories. In this example, under normal circumstances, the mIoU index value of the model on the verification set will reach 0.80 or more. An example of the displayed information is shown below. The **mIoU=0.8526** in the third row is mIoU.
- **Kappa coefficient**: an index used for consistency testing, which can be used to measure the effect of classification. The calculation of the kappa coefficient is based on the confusion matrix, with a value between -1 and 1, usually greater than 0. The formula is as follows, P0P_0*P*0 is the accuracy of the classifier, and PeP_e*P**e* is the accuracy of the random classifier. The higher the Kappa coefficient, the better the model quality.

<a href="https://www.codecogs.com/eqnedit.php?latex=Kappa=&space;\frac{P_0-P_e}{1-P_e}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Kappa=&space;\frac{P_0-P_e}{1-P_e}" title="Kappa= \frac{P_0-P_e}{1-P_e}" /></a>

With the running of the evaluation script, the final printed evaluation log is as follows.

```
...
2021-01-13 16:41:29 [INFO]	Start evaluating (total_samples=76, total_iters=76)...
76/76 [==============================] - 2s 30ms/step - batch_cost: 0.0268 - reader cost: 1.7656e-
2021-01-13 16:41:31 [INFO]	[EVAL] #Images=76 mIoU=0.8526 Acc=0.9942 Kappa=0.8283
2021-01-13 16:41:31 [INFO]	[EVAL] Class IoU:
[0.9941 0.7112]
2021-01-13 16:41:31 [INFO]	[EVAL] Class Acc:
[0.9959 0.8886]
```

### 2.Evaluation and Prediction under **API** Approach

#### Evaluating

Construct Model
```
from paddleseg.models import BiSeNetV2
model = BiSeNetV2(num_classes=2,
                 lambd=0.25,
                 align_corners=False,
                 pretrained=None)
```

Load Model Parameters

```
model_path = 'output/best_model/model.pdparams'# Path of best model
if model_path:
    para_state_dict = paddle.load(model_path)  
    model.set_dict(para_state_dict)            # Load parameters
    print('Loaded trained params of model successfully')
else:
    raise ValueError('The model_path is wrong: {}'.format(model_path))
```

Construct Validation Dataset

```
# Define transforms for verification
import paddleseg.transforms as T
transforms = [
    T.Resize(target_size=(512, 512)),
    T.Normalize()
]

# Construct validation dataset
from paddleseg.datasets import OpticDiscSeg
val_dataset = OpticDiscSeg(
    dataset_root='data/optic_disc_seg',
    transforms=transforms,
    mode='val'
)
```
**Evaluate** API Parameter Analysis


```
paddleseg.core.evaluate(
                        model,
                        eval_dataset,
                        aug_eval=False,
                        scales=1.0,  
                        flip_horizontal=True,
                        flip_vertical=False,
                        is_slide=False,
                        stride=None,
                        crop_size=None,
                        num_workers=0  
)
```

- Parameters

| Parameter          | Types          | Effection                                                 | Is Required | Default |
| --------------- | ----------------- | ---------------------------------------------------- | ---------- | ------ |
| model           | nn.Layer          | Segmentation model                            | Yes         | -        |
| eval_dataset    | paddle.io.Dataset | Validation dataSet                                        | Yes         | -      |
| aug_eval        | bool              | Whether to use data augmentation                 | No         | False  |
| scales          | list/float        | Set the zoom factor, take effect when aug_pred is True                   | No         | 1.0      |
| flip_horizontal | bool              | Whether to use horizontal flip, take effect when `aug_eval` is True      | No         | True     |
| flip_vertical   | bool              | Whether to use vertical flip, take effect when `aug_eval` is True        | No         | False    |
| is_slide        | bool              | Whether to evaluate through a sliding window                             | No         | False    |
| stride          | tuple/list        | Set the width and height of the sliding window, effective when `is_slide` is True       | No         | None     |
| crop_size       | tuple/list        | Set the width and height of the crop of the sliding window, which takes effect when `is_slide` is True | No         | None     |
| num_workers     | int               | Multi-threaded data loading                                       | No         | 0      |



**Note** If you want to improve the memory utilization, you can increase the setting of num_workers appropriately to prevent the GPU from waiting during work.


Import the API interface and start the evaluation

```
from paddleseg.core import evaluate
evaluate(
        model,
        val_dataset #paddle.io.Dataset，验证集DataSet
)
```

Multi-scale , flip evaluation

```
evaluate(
        model,
        val_dataset,
        aug_eval=True,  #是否使用数据增强
        scales=[0.75, 1.0, 1.25],  #缩放因子
        flip_horizontal=True)  #是否水平翻转
```

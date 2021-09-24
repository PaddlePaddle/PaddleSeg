English|[简体中文](predict_cn.md)

# Prediction

In addition to analyzing the `IOU`, `ACC` and `Kappa`, we can also check the segmentation effect of some specific samples, and inspire further optimization ideas from Bad Case.

The `predict.py` script is specially used to visualize prediction cases. The command format is as follows:

```
python predict.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --model_path output/iter_1000/model.pdparams \
       --image_path dataset/optic_disc_seg/JPEGImages/H0003.jpg \
       --save_dir output/result
```

Among them, `image_path` can also be a directory. At this time, all the images in the directory will be predicted and the visualization results will be saved.

Similarly, you can use `--aug_pred` to turn on multi-scale flip prediction, and `--is_slide` to turn on sliding window prediction.


## 1.Prepare Dataset

- When performing prediction, only the original image is needed. You should prepare the contents of `test.txt` as follows:
    ```
    images/image1.jpg
    images/image2.jpg
    ...
    ```

- When calling `predict.py` for visualization, annotated images can be included in the file list. When predicting, the model will automatically ignore the annotated images given in the file list. Therefore, you can also directly use the training and validating datasets to do predictions. In other words, if the content of your `train.txt` is as follows：
    ```
    images/image1.jpg labels/label1.png
    images/image2.jpg labels/label2.png
    ...
    ```

* At this point, you can specify `image_list` as `train.txt` and `image_dir` as the directory where the training data is located when predicting. The robustness of PaddleSeg allows you to do this, and the output will be the prediction result of the **original training data**.

## 2.API
Parameter Analysis of Forecast API 

```
paddleseg.core.predict(
                    model,
                    model_path,
                    transforms,
                    image_list,
                    image_dir=None,
                    save_dir='output',
                    aug_pred=False,
                    scales=1.0,
                    flip_horizontal=True,
                    flip_vertical=False,
                    is_slide=False,
                    stride=None,
                    crop_size=None,
                    custom_color=None
)
```

- Parameters

| Parameter          | Type          | Effection                                                 | Is Required | Default |
| --------------- | ----------------- | ---------------------------------------------------- | ---------- | -------- |
| model           | nn.Layer          | Segmentation model                            | Yes         | -        |
| model_path      | str               | The path of parameters in best model          | Yes         | -        |
| transforms      | transform.Compose | Preprocess the input image                    | Yes         | -        |
| image_list      | list              | List of image paths to be predicted             | Yes         | -        |
| image_dir       | str               | The directory of the image path to be predicted     | No         | None     |
| save_dir        | str               | Output directory                                         | No         | 'output' |
| aug_pred        | bool              | Whether to use multi-scale and flip augmentation for prediction          | No         | False    |
| scales          | list/float        | Set the zoom factor, take effect when aug_pred is True                   | No         | 1.0      |
| flip_horizontal | bool              | Whether to use horizontal flip, take effect when `aug_eval` is True      | No         | True     |
| flip_vertical   | bool              | Whether to use vertical flip, take effect when `aug_eval` is True        | No         | False    |
| is_slide        | bool              | Whether to evaluate through a sliding window                             | No         | False    |
| stride          | tuple/list        | Set the width and height of the sliding window, effective when `is_slide` is True       | No         | None     |
| crop_size       | tuple/list        | Set the width and height of the crop of the sliding window, which takes effect when `is_slide` is True | No         | None     |
| custom_color    | list              | Set custom segmentation prediction colors,len(custom_color) = 3 * (pixel classes)  | No        | Default color map |

Import the API interface and start predicting.

```
from paddleseg.core import predict
predict(
        model,
        model_path='output/best_model/model.pdparams',# Model path
        transforms=transforms, # Transform.Compose， Preprocess the input image
        image_list=image_list, # List of image paths to be predicted。
        image_dir=image_dir, # The directory where the picture to be predicted is located
        save_dir='output/results' # Output path
    )
```

## 3.Instruction of File Structure
If you don't specify the output location, `added_prediction` and `pseudo_color_prediction` will be generated under the default folder `output/results`, which store the results of the pseudo map and blended prediction respectively.

    output/result
        |
        |--added_prediction
        |  |--image1.jpg
        |  |--image2.jpg
        |  |--...
        |
        |--pseudo_color_prediction
        |  |--image1.jpg
        |  |--image2.jpg
        |  |--...




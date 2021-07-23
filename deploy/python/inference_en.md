English | [简体中文](inference_cn.md)
# Local inference deployment

## 1. Instruction

This solution aims to provide a Python prediction deployment solution for the PaddlePaddle cross-platform image segmentation model as a reference. Users can integrate the model into their own services through a certain configuration and add a small amount of code to complete the task of image segmentation.


## 2. Pre-preparation

Please refer to [document](../../export/export/model_export.md) to export your model, or click to download our [sample model](https://paddleseg.bj.bcebos.com/dygraph/demo/bisenet_demo_model.tar.gz) is used for testing.

Then prepare a picture for the test effect, we provide a [picture] (https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png) in the cityscapes verification set to see the effect, if your model is trained by other data sets, please prepare test images by yourself.


## 3. Prediction

Enter the following command in the terminal to make predictions:
```shell
python deploy/python/infer.py --config /path/to/deploy.yaml --image_path
```

Parameters:
|Parameter|Effection|Is required|Default|
|-|-|-|-|
|config|**Configuration file generated when exporting the model**, instead of the configuration file in the configs directory|Yes|-|
|image_path|The path or directory of the test image.|Yes|-|
|use_trt|Whether to enable TensorRT to accelerate prediction.|No|No|
|use_int8|Whether to run in int8 mode when starting TensorRT prediction.|No|No|
|batch_size|Batch sizein single card.|No|The value specified in the configuration file.|
|save_dir|The directory of prediction results.|No|output|
|with_argmax|Perform argmax operation on the prediction results.|No|No|

*The test sample and prediction result*
![cityscape_predict_demo.png](../../../docs/images/cityscapes_predict_demo.png)

*Note:*
*1. When using a quantitative model to predict, you need to turn on TensorRT prediction and int8 prediction at the same time to have an acceleration effect*
*2. To use TensorRT, you need to use the Paddle library that supports the TRT function, please refer to [Appendix](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#whl-release) to download the corresponding PaddlePaddle installation Package, or refer to [source code compilation](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/fromsource.html) to compile by yourself*

## 4. Calculate inference speed
After deploying the exported model, you may be very concerned about the inference speed of the deployed model. PaddleSeg provides a method to test the inference speed of the model. The following will introduce how to test the inference speed on your machine.
* First, prepare the test data set (it is recommended to use more than 100 original images, do not use annotated images)
* Execute the following commands in the terminal：
```shell
python deploy/python/infer_timer.py \
--config output/BisenetV2/deploy.yaml \
--image_path /path/to/data_dir \
```
> Among them, `/path/to/data_dir` refers to the path where the image is stored, please modify it according to the actual situation.
> `data_dir` could be a single image or a directory.
> When you do this, we recommend you using a directory containing multiple images as `data_dir` to get a more reliable average value of the inference speed.

### Note
* 1、If the `image_num` which you specify is greater than the total number of images stored in `image_path`, we will use the actual number of images.
* 2、This method only calculates the inference speed and does not involve obtaining segmentation results. If you want to obtain the segmentation results obtained by inference, please run as indicated in `3`.
* 3、This method only calculates the inferencing time, not include pre-processing time and post-processing time.
* 4、The result is related to various factors such as `machine configuration`, `memory`, `image resolution`, etc. The results are for reference only.

Parameter instruction:
|Parameter|Effection|Is required|Default|
|-|-|-|-|
|config|**Configuration file generated when exporting the model**, instead of the configuration file in the configs directory|Yes|-|
|image_path|The path or directory of the test image.|Yes|-|
|image_num|The number of test images, used to calculate the average.|No|100|
|use_trt|Whether to enable TensorRT to accelerate prediction.|No|No|
|use_int8|Whether to run in int8 mode when starting TensorRT prediction.|No|No|
|batch_size|Batch sizein single card.|No|The value specified in the configuration file.|
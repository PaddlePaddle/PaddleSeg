# Local Inference deployment

## 1. Description

This document describes deploying segmentation models on the server side using the Python interface for Paddle Inference. With a certain configuration and a small amount of code, you can integrate the model into your own service to complete the task of image segmentation.

Paddle Inference's [official website document](https://paddleinference.paddlepaddle.org.cn/product_introduction/summary.html) introduces the steps to deploy the model, various API interfaces, examples, etc. You can use it according to your actual needs .

## 2. Preliminary preparation

Please use the [Model Export Tool](../../model_export.md) to export your model, or click to download our [Sample Model](https://paddleseg.bj.bcebos.com/dygraph/demo/bisenet_demo_model.tar.gz) for testing.

Then prepare a test image for the test effect, we provide a [image](https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png) in the validation set of cityscapes for demonstration effect, if Your model is trained using another dataset, please prepare test images by yourself.

## 3. Prediction

Enter the following command in the terminal to make predictions:
```shell
python deploy/python/infer.py --config /path/to/deploy.yaml --image_path
````

The parameter descriptions are as follows:
|Parameter name|Purpose|Required option|Default value|
|-|-|-|-|
|config|**The configuration file generated when exporting the model**, not the configuration file in the configs directory|Yes|-|
|image_path|The path or directory of the predicted image|is|-|
|use_cpu|Whether to use X86 CPU prediction, the default is to use GPU prediction|No|No|
|use_trt|Whether to enable TensorRT to speed up prediction|No|No|
|use_int8|Whether to run in int8 mode when starting TensorRT prediction|no|no|
|use_mkldnn|Whether to enable MKLDNN for accelerated prediction|No|No|
|batch_size|single card batch size|no|specified value in configuration file|
|save_dir|Directory where prediction results are saved|no |output|
|with_argmax| Perform argmax operation on the prediction result|No|No|

*Test examples and predicted results are as follows*
![cityscape_predict_demo.png](../../images/cityscapes_predict_demo.png)

**Notice**

1. When using quantitative model prediction, both TensorRT prediction and int8 prediction need to be turned on to have the acceleration effect

2. To use TensorRT, you need to use the Paddle library that supports the TRT function. Please refer to [Appendix](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#whl-release) to download the corresponding PaddlePaddle installation package, or refer to [source code compilation](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/fromsource.html) to compile it yourself.

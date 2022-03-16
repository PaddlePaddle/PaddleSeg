English | [简体中文](README_CN.md)

# Human Matting Android Demo
Based on [PaddleSeg](https://github.com/paddlepaddle/paddleseg/tree/develop) [MODNet](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/contrib/Matting) algorithm to realise human matting（Android demo).

You can directly download and install the example project [apk](https://paddleseg.bj.bcebos.com/matting/models/deploy/app-debug.apk) to experience。

## 1. Results Exhibition
<div align="center">
<img src=https://user-images.githubusercontent.com/14087480/141890516-6aad4691-9ab3-4baf-99e5-f1afa1b21281.png  width="50%">

</div>


## 2. Android Demo Instructions

### 2.1 Reruirements
* Android Studio 3.4；
* Android mobile phone；

### 2.2 Installation
* open Android Studio and on "Welcome to Android Studio" window, click "Open an existing Android Studio project". In the path selection window that is displayed, select the folder corresponding to the Android Demo. Then click the "Open" button in the lower right corner to import the project. The Lite prediction library required by demo will be automatically downloaded during the construction process.
* Connect Android phone via USB;
* After loading the project, click the Run->Run 'App' button on the menu bar, Select the connected Android device in the "Select Deployment Target" window, and then click the "OK" button;

*Note：this Android demo is based on [Paddle-Lite](https://paddlelite.paddlepaddle.org.cn/)，PaddleLite version is 2.8.0*

### 2.3 Prediction
* In human matting demo, a human image will be loaded by default, and the CPU prediction result and prediction delay will be given below the image;
* In the human matting demo, you can also load test images from the album or camera by clicking the "Open local album" and "Open camera to take photos" buttons in the upper right corner, and then perform prediction.

*Note：When taking a photo in demo, the photo will be compressed automatically. If you want to test the effect of the original photo, you can use the mobile phone camera to take a photo and open it from the album for prediction.*

## 3. Secondary Development
The inference library or model can be updated according to the need for secondary development. The updated model can be divided into two steps: model export and model transformation.

### 3.1 Update Inference Library
[Paddle-Lite website](https://paddlelite.paddlepaddle.org.cn/) probides a pre-compiled version of Android inference library, which can also be compiled by referring to the official website.

The Paddle-Lite inference library on Android mainly contains three files:

* PaddlePredictor.jar；
* arm64-v8a/libpaddle_lite_jni.so；
* armeabi-v7a/libpaddle_lite_jni.so；

Two methods will be introduced in the following：

* Use a precompiled version of the inference library. Latest precompiled file reference：[release](https://github.com/PaddlePaddle/Paddle-Lite/releases/). This demo uses the [version](https://paddlelite-demo.bj.bcebos.com/libs/android/paddle_lite_libs_v2_8_0.tar.gz)

	Uncompress the above files and the PaddlePredictor.jar is in java/PaddlePredictor.jar;

	The so file about arm64-v8a is in java/libs/arm64-v8a;

	The so file about armeabi-v7a is in java/libs/armeabi-v7a;

* Manually compile the paddle-Lite inference library
Development environment preparation and compilation methods refer to [paddle-Lite source code compilation](https://paddle-lite.readthedocs.io/zh/release-v2.8/source_compile/compile_env.html).

Prepare the above documents, then refer [java_api](https://paddle-lite.readthedocs.io/zh/release-v2.8/api_reference/java_api_doc.html)to have a inference on Android. Refer to the documentation in the Update Inference Library section of [Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo) for details on how to use the inference library.

### 3.2 Model Export
This demo uses the MODNet with HRNet_W18 backbone to perform human matting. Please refer to official websit to get model [training tutorial](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/contrib/Matting). There are 3 models provided with different backbones: MobileNetV2、ResNet50_vd and HRNet_W18. This Android demo considers the accuracy and speed, using HRNet_W18 as the Backone. The trained dynamic graph model can be downloaded directly from the official website for algorithm verification.

In order to be able to infer on Android phones, the dynamic graph model needs to be exported as a static graph model, and the input size of the image should be fixed when exporting.

First, update the [PaddleSeg](https://github.com/paddlepaddle/paddleseg/tree/develop) repository. Then `cd` to the `PaddleSeg/contrib/Matting` directory. Then put the downloaded modnet-hrnet_w18.pdparams (traing by youself is ok) on current directory（`PaddleSeg/contrib/Matting`). After that, fix the config file `configs/modnet_mobilenetv2.yml`(note: hrnet18 is used, but the config file `modnet_hrnet_w18.yml` is based on `modnet_mobilenetv2.yml`), where,modify the val_dataset field as follows:

``` yml
val_dataset:
  type: MattingDataset
  dataset_root: data/PPM-100
  val_file: val.txt
  transforms:
    - type: LoadImages
    - type: ResizeByShort
      short_size: 256
    - type: ResizeToIntMult
      mult_int: 32
    - type: Normalize
  mode: val
  get_trimap: False
```
In the above modification, pay special attention to the short_size: 256 field which directly determines the size of our final inferential image. If the value of this field is set too small, the prediction accuracy will be affected; if the value is set too high, the inference speed of the phone will be affected (or even the phone will crash due to performance problems). In practical testing, this field is set to 256 for HRnet18.

After modifying the configuration file, run the following command to export the static graph:
``` shell
python export.py \
    --config configs/modnet/modnet_hrnet_w18.yml \
    --model_path modnet-hrnet_w18.pdparams \
    --save_dir output
```

After the conversion, the `output` folder will be generated in the current directory, and the files in the folder are the static graph files.

### 3.3 Model Conversion

#### 3.3.1 Model Conversion Tool
Once you have the static diagram model and parameter files exported from PaddleSeg ready, you need to optimize the model using the opt provided with Paddle-Lite and convert to the file format supported by Paddle-Lite.

Firstly, install PaddleLite:

``` shell
pip install paddlelite==2.8.0
```

Then use the following Python script to convert:

``` python
# Reference the Paddlelite inference library
from paddlelite.lite import *

# 1. Create opt instance
opt=Opt()

# 2. Specify the static model path
opt.set_model_file('./output/model.pdmodel')
opt.set_param_file('./output/model.pdiparams')

# 3. Specify conversion type: arm, x86, opencl or npu
opt.set_valid_places("arm")
# 4. Specifies the model transformation type: naive_buffer or protobuf
opt.set_model_type("naive_buffer")
# 5. Address of output model
opt.set_optimize_out("./output/hrnet_w18")
# 6. Perform model optimization
opt.run()
```

After conversion, the `hrnet_w18.nb` file will be generated in the `output` directory.

#### 3.3.2 Update Model
Using the optimized `. Nb ` file to replace the file in `app/SRC/main/assets/image_matting/models/modnet` in android applications.

Then change the image input size in the project: open the string.xml file and follow the under example:
``` xml
<string name="INPUT_SHAPE_DEFAULT">1,3,256,256</string>
```
1,3,256,256 represent the corresponding batchsize, channel, height and width respectively. Generally, height and width are modified to the size which set during model export.

The entire android demo is implemented in Java, without embedded C++ code, which is relatively easy to build and execute. In the future, you can also move this demo to Java Web projects for human matting in the Web.

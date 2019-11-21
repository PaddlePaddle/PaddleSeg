# 安卓端人像分割

## 1.介绍
基于Paddle-Lite v2.0.0实现的人像分割Android示例`模型链接`，基于`数据`训练。以下第二节介绍如何使用demo，后面几章节介绍如何将PaddleSeg的Model部署到安卓设备。

## 2.安卓端Demo使用

### 2.1 要求
*  Android Studio 3.4；
* Android手机或开发版；

### 2.2 安装
* 打开Android Studio，在"Welcome to Android Studio"窗口点击"Open an existing Android Studio project"，在弹出的路径选择窗口中进入"PaddleLite-android-demo"目录，然后点击右下角的"Open"按钮即可导入工程
* 通过USB连接Android手机或开发板；
* 载入工程后，点击菜单栏的Run->Run 'App'按钮，在弹出的"Select Deployment Target"窗口选择已经连接的Android设备，然后点击"OK"按钮；
* 手机上会出现Demo的主界面，选择"Image Segmentation"图标，进入的人像分割示例程序；
* 在人像分割Demo中，默认会载入一张人像图像，并会在图像下方给出CPU的预测结果；
* 在人像分割Demo中，你还可以通过上方的"Gallery"和"Take Photo"按钮从相册或相机中加载测试图像；

### 2.3 更新model
将第4.2节优化好的model.nb和param.nb文件，替app/src/main/assets/image_segmentation/
models/deeplab_mobilenet_for_cpu下面的文件即可。

### 2.4 其他
此安卓demo基于[Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo)开发，更多的细节请参考该repo。

### 2.5 效果展示
***预留***

## 3.模型导出
[模型导出](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.2.0/docs/model_export.md)

## 4.模型部署
为了支持PaddleSeg模型在移动端的部署，需要准备Paddle-Lite预测库和模型转换工具。

### 4.1预测库
Paddle-Lite的编译目前支持Docker，Linux和Mac OS开发环境，建议使用Docker开发环境，以免存在各种依赖问题，同时也提供了预编译版本的预测库，准备Paddle-Lite在安卓端的预测库，主要包括三个文件：

* PaddlePredictor.jar；
* arm64-v8a/libpaddle_lite_jni.so；
*  armeabi-v7a/libpaddle_lite_jni.so；

下面分别介绍两种方法：

* 使用预编译版本的预测库，最新的预编译文件参考：[release](https://github.com/PaddlePaddle/Paddle-Lite/releases/)，此demo使用的版本：

    * arm64-v8a: [inference_lite_lib.android.armv8](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.0.0/inference_lite_lib.android.armv8.gcc.c++_shared.with_extra.full_publish.tar.gz) ；
    
    * armeabi-v7a: [inference_lite_lib.android.armv7](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.0.0/inference_lite_lib.android.armv7.gcc.c++_shared.with_extra.full_publish.tar.gz) ；

    解压上面两个文件，PaddlePredictor.jar位于任一文件夹：inference_lite_lib.android.xxx/java/jar/PaddlePredictor.jar；
        
    解压上述inference_lite_lib.android.armv8文件，arm64-v8a/libpaddle_lite_jni.so位于：inference_lite_lib.android.armv8/java/so/libpaddle_lite_jni.so；
    
    解压上述inference_lite_lib.android.armv7文件，armeabi-v7a/libpaddle_lite_jni.so位于：inference_lite_lib.android.armv7/java/so/libpaddle_lite_jni.so；

* 手动编译Paddle-Lite预测库
开发环境的准备和编译方法参考：[Paddle-Lite源码编译](https://paddlepaddle.github.io/Paddle-Lite/v2.0.0/source_compile/)。

准备好上述文件，即可参考[java_api](https://paddlepaddle.github.io/Paddle-Lite/v2.0.0/java_api_doc/)在安卓端进行推理。具体使用预测库的方法可参考[Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo)。

### 4.2模型转换

准备好预测库，以及PaddleSeg导出来的模型和参数文件后，需要使用Paddle-Lite提供的model_optimize_tool对模型进行优化，并转换成Paddle-Lite支持的文件格式，这里有两种方式来实现：

* 使用预编译版本的model_optimize_tool，最新的预编译文件参考[release](https://github.com/PaddlePaddle/Paddle-Lite/releases/)，此demo使用的版本为[model_optimize_tool](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.0.0/model_optimize_tool) ；

*注意：如果运行失败，请在上一节准备好的开发环境中使用model_optimize_tool*

* 手动编译model_optimize_tool
详细的模型转换方法参考paddlelite提供的官方文档：[模型转化方法](https://paddlepaddle.github.io/Paddle-Lite/v2.0.0/model_optimize_tool/)，从PaddleSeg里面导出来的模型使用如下指令即可导出model.nb和param.nb文件。

    ```
    ./model_optimize_tool \
        --model_dir=<model_param_dir> \
        --model_file=<model_path> \
        --param_file=<param_path> \
        --optimize_out_type=naive_buffer \
        --optimize_out=<output_optimize_model_dir> \
        --valid_targets=arm \
    ```

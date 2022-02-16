简体中文 | [English](README.md)

# Human Matting Android Demo
基于[PaddleSeg](https://github.com/paddlepaddle/paddleseg/tree/develop)的[MODNet](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/contrib/Matting)算法实现人像抠图（安卓版demo）。

可以直接下载安装本示例工程中的[apk](https://paddleseg.bj.bcebos.com/matting/models/deploy/app-debug.apk)进行体验。

## 1. 效果展示
<div align="center">
<img src=https://user-images.githubusercontent.com/14087480/141890516-6aad4691-9ab3-4baf-99e5-f1afa1b21281.png  width="50%">

</div>


## 2. 安卓Demo使用说明

### 2.1 要求
* Android Studio 3.4；
* Android手机；

### 2.2 一键安装
* 打开Android Studio，在"Welcome to Android Studio"窗口点击"Open an existing Android Studio project"，在弹出的路径选择窗口中选择本安卓demo对应的文件夹，然后点击右下角的"Open"按钮即可导入工程，构建工程的过程中会自动下载demo需要的Lite预测库；
* 通过USB连接Android手机；
* 载入工程后，点击菜单栏的Run->Run 'App'按钮，在弹出的"Select Deployment Target"窗口选择已经连接的Android设备，然后点击"OK"按钮；

*注：此安卓demo基于[Paddle-Lite](https://paddlelite.paddlepaddle.org.cn/)开发，PaddleLite版本为2.8.0。*

### 2.3 预测
* 在人像抠图Demo中，默认会载入一张人像图像，并会在图像下方给出CPU的预测结果和预测延时；
* 在人像抠图Demo中，你还可以通过右上角的"打开本地相册"和"打开摄像头拍照"按钮分别从相册或相机中加载测试图像然后进行预测推理；

*注意：demo中拍照时照片会自动压缩，想测试拍照原图效果，可使用手机相机拍照后从相册中打开进行预测。*

## 3. 二次开发
可按需要更新预测库或模型进行二次开发，其中更新模型分为模型导出和模型转换两个步骤。

### 3.1 更新预测库
[Paddle-Lite官网](https://paddlelite.paddlepaddle.org.cn/)提供了预编译版本的安卓预测库，也可以参考官网自行编译。

Paddle-Lite在安卓端的预测库主要包括三个文件：

* PaddlePredictor.jar；
* arm64-v8a/libpaddle_lite_jni.so；
* armeabi-v7a/libpaddle_lite_jni.so；

下面分别介绍两种方法：

* 使用预编译版本的预测库，最新的预编译文件参考：[release](https://github.com/PaddlePaddle/Paddle-Lite/releases/)，此demo使用的[版本](https://paddlelite-demo.bj.bcebos.com/libs/android/paddle_lite_libs_v2_8_0.tar.gz)

	解压上面文件，PaddlePredictor.jar位于：java/PaddlePredictor.jar；

	arm64-v8a相关so位于：java/libs/arm64-v8a；

	armeabi-v7a相关so位于：java/libs/armeabi-v7a；

* 手动编译Paddle-Lite预测库
开发环境的准备和编译方法参考：[Paddle-Lite源码编译](https://paddle-lite.readthedocs.io/zh/release-v2.8/source_compile/compile_env.html)。

准备好上述文件，即可参考[java_api](https://paddle-lite.readthedocs.io/zh/release-v2.8/api_reference/java_api_doc.html)在安卓端进行推理。具体使用预测库的方法可参考[Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo)中更新预测库部分的文档。

### 3.2 模型导出
此demo的人像抠图采用Backbone为HRNet_W18的MODNet模型，模型[训练教程](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/contrib/Matting)请参考官网，官网提供了3种不同性能的Backone：MobileNetV2、ResNet50_vd和HRNet_W18。本安卓demo综合考虑精度和速度要求，采用了HRNet_W18作为Backone。可以直接从官网下载训练好的动态图模型进行算法验证。

为了能够在安卓手机上进行推理，需要将动态图模型导出为静态图模型，导出时固定图像输入尺寸即可。

首先git最新的[PaddleSeg](https://github.com/paddlepaddle/paddleseg/tree/develop)项目，然后cd进入到PaddleSeg/contrib/Matting目录。将下载下来的modnet-hrnet_w18.pdparams动态图模型文件（也可以自行训练得到）放置在当前文件夹（PaddleSeg/contrib/Matting）下面。然后修改配置文件 configs/modnet_mobilenetv2.yml(注意：虽然采用hrnet18模型，但是该模型依赖的配置文件modnet_hrnet_w18.yml本身依赖modnet_mobilenetv2.yml),修改其中的val_dataset字段如下：

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
上述修改中尤其注意short_size: 256这个字段，这个值直接决定我们最终的推理图像采用的尺寸大小。这个字段值设置太小会影响预测精度，设置太大会影响手机推理速度（甚至造成手机因性能问题无法完成推理而崩溃）。经过实际测试，对于hrnet18，该字段设置为256较好。

完成配置文件修改后，采用下面的命令进行静态图导出：
``` shell
python export.py \
    --config configs/modnet/modnet_hrnet_w18.yml \
    --model_path modnet-hrnet_w18.pdparams \
    --save_dir output
```

转换完成后在当前目录下会生成output文件夹，该文件夹中的文件即为转出来的静态图文件。

### 3.3 模型转换

#### 3.3.1 模型转换工具
准备好PaddleSeg导出来的静态图模型和参数文件后，需要使用Paddle-Lite提供的opt对模型进行优化，并转换成Paddle-Lite支持的文件格式。

首先安装PaddleLite：

``` shell
pip install paddlelite==2.8.0
```

然后使用下面的python脚本进行转换：

``` python
# 引用Paddlelite预测库
from paddlelite.lite import *

# 1. 创建opt实例
opt=Opt()

# 2. 指定静态模型路径
opt.set_model_file('./output/model.pdmodel')
opt.set_param_file('./output/model.pdiparams')

# 3. 指定转化类型： arm、x86、opencl、npu
opt.set_valid_places("arm")
# 4. 指定模型转化类型： naive_buffer、protobuf
opt.set_model_type("naive_buffer")
# 5. 输出模型地址
opt.set_optimize_out("./output/hrnet_w18")
# 6. 执行模型优化
opt.run()
```

转换完成后在output目录下会生成对应的hrnet_w18.nb文件。

#### 3.3.2 更新模型
将优化好的`.nb`文件，替换安卓程序中的 app/src/main/assets/image_matting/
models/modnet下面的文件即可。

然后在工程中修改图像输入尺寸：打开string.xml文件，修改示例如下：
``` xml
<string name="INPUT_SHAPE_DEFAULT">1,3,256,256</string>
```
1,3,256,256分别表示图像对应的batchsize、channel、height、width，我们一般修改height和width即可，这里的height和width需要和静态图导出时设置的尺寸一致。

整个安卓demo采用java实现，没有内嵌C++代码，构建和执行比较简单。未来也可以将本demo移植到java web项目中实现web版人像抠图。

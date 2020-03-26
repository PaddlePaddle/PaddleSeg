# 视频实时人像分割模型Windows平台C++预测部署

## 1. 系统和软件依赖

### 1.1 基础依赖

- Windows 10 / Windows Server 2016+ (其它平台未测试)
- Visual Studio 2019 (社区版或专业版均可)
- CUDA 9.0 / 10.0 + CUDNN 7.3+ (不支持9.1/10.1版本的CUDA)

### 1.2 下载OpenCV并设置环境变量

- 在OpenCV官网下载适用于Windows平台的3.4.6版本: [点击下载](https://sourceforge.net/projects/opencvlibrary/files/3.4.6/opencv-3.4.6-vc14_vc15.exe/download)
- 运行下载的可执行文件，将OpenCV解压至合适目录，这里以解压到`D:\projects\opencv`为例
- 把OpenCV动态库加入到系统环境变量
   - 此电脑(我的电脑)->属性->高级系统设置->环境变量
   - 在系统变量中找到Path（如没有，自行创建），并双击编辑
   - 新建，将opencv路径填入并保存，如D:\projects\opencv\build\x64\vc14\bin

**注意:** `OpenCV`的解压目录后续将做为编译配置项使用，所以请放置合适的目录中。

### 1.3 下载PaddlePaddle C++ 预测库

`PaddlePaddle` **C++ 预测库** 主要分为`CPU`和`GPU`版本， 其中`GPU版本`提供`CUDA 9.0` 和 `CUDA 10.0` 支持。

常用的版本如下：

|  版本   | 链接  |
|  ----  | ----  |
| CPU+MKL版  | [fluid_inference_install_dir.zip](https://paddle-wheel.bj.bcebos.com/1.6.3/win-infer/mkl/cpu/fluid_inference_install_dir.zip) |
| CUDA9.0+MKL 版  | [fluid_inference_install_dir.zip](https://paddle-wheel.bj.bcebos.com/1.6.3/win-infer/mkl/post97/fluid_inference_install_dir.zip) |
| CUDA10.0+MKL 版 | [fluid_inference_install_dir.zip](https://paddle-wheel.bj.bcebos.com/1.6.3/win-infer/mkl/post107/fluid_inference_install_dir.zip) |

更多不同平台的可用预测库版本，请[点击查看](https://paddlepaddle.org.cn/documentation/docs/zh/advanced_usage/deploy/inference/windows_cpp_inference.html) 选择适合你的版本。


下载并解压, 解压后的 `fluid_inference`目录包含的内容：
```
fluid_inference_install_dir
├── paddle # paddle核心库和头文件
|
├── third_party # 第三方依赖库和头文件
|
└── version.txt # 版本和编译信息
```

**注意：** 这里的`fluid_inference_install_dir` 目录所在路径，将用于后面的编译参数设置，请放置在合适的位置。

## 2. Visual Studio 2019 编译

- 2.1 打开Visual Studio 2019 Community，点击`继续但无需代码`， 如下图：
![step2.1](https://paddleseg.bj.bcebos.com/inference/vs2019_step1.png)

- 2.2 点击 `文件`->`打开`->`CMake`, 如下图：
![step2.2](https://paddleseg.bj.bcebos.com/inference/vs2019_step2.png)  

- 2.3 选择本项目根目录`CMakeList.txt`文件打开， 如下图：
![step2.3](https://paddleseg.bj.bcebos.com/deploy/docs/vs2019_step2.3.png)

- 2.4 点击：`项目`->`PaddleMaskDetector的CMake设置`
![step2.4](https://paddleseg.bj.bcebos.com/deploy/docs/vs2019_step2.4.png)

- 2.5 点击浏览设置`OPENCV_DIR`, `CUDA_LIB` 和 `PADDLE_DIR` 3个编译依赖库的位置, 设置完成后点击`保存并生成CMake缓存并加载变量`
![step2.5](https://paddleseg.bj.bcebos.com/inference/vs2019_step5.png)

- 2.6 点击`生成`->`全部生成` 编译项目
![step2.6](https://paddleseg.bj.bcebos.com/inference/vs2019_step6.png)

## 3. 运行程序

成功编译后， 产出的可执行文件在项目子目录`out\build\x64-Release`目录， 按以下步骤运行代码：

- 打开`cmd`切换至该目录
- 运行以下命令传入模型路径与测试视频

```shell
main.exe ./models/ ./data/test.avi
```
第一个参数即人像分割预测模型的路径，第二个参数即要预测的视频。

运行后，预测结果保存在文件`result.avi`中。

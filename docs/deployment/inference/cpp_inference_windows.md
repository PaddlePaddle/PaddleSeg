English | [简体中文](cpp_inference_windows_cn.md)

# Paddle Inference Deployment on Windows（C++）

## 1. Description

This document introduces an example of deploying a segmentation model on Windows using Paddle Inference's C++ interface. The main steps include:
* Prepare the environment
* Prepare models and pictures
* Compile and execute

PaddlePaddle provides multiple prediction engine deployment models (as shown in the figure below) for different scenarios. For details, please refer to [document](https://www.paddlepaddle.org.cn/inference/v2.3/product_introduction/summary.html)。

![inference_ecosystem](https://user-images.githubusercontent.com/52520497/130720374-26947102-93ec-41e2-8207-38081dcc27aa.png)

## 2. Prepare the environment

### 2.1 Prepare the basic environment

The basic environment requirements for model deployment are as follows:
* Visual Studio 2019 (According to the VS version used by Paddle Inference C++ prediction library, please refer to [C++ binary compatibility between Visual Studio versions](https://docs.microsoft.com/en-us/cpp/porting/binary-compat-2015-2017?view=vs-2019) )
* CUDA / CUDNN / TensorRT(Only required when using GPU version of prediction library)
* CMake 3.0+ [CMake download](https://cmake.org/download/)

All the following examples are demonstrated with the working directory `D:\projects`.

### 2.2 Prepare CUDA/CUDNN/TensorRT environment

The model deployment environment and the libraries to be prepared are shown in the following table:

| Deployment environment |      Libraries      |
|:----------------------:|:-------------------:|
|          CPU           |          -          |
|          GPU           |     CUDA/CUDNN      |
|        GPU_TRT         | CUDA/CUDNN/TensorRT |

Users who use GPU for inference need to prepare CUDA and CUDNN according to the following instructions. Users who use CPU for inference can skip.  

CUDA installation, please refer to [Official Tutorial](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#verify-you-have-cuda-enabled-system).  
The default installation path of CUDA is `C:\Program Files\NVIDIA GPU Computing Toolkit`. Add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\Vx.y\bin`to the environment variable.


CUDNN installation, please refer to [Official Tutorial](https://docs.nvidia.com/deeplearning/cudnn/install-guide/#install-windows).  
Copy the files in the `bin`, `include`, and `lib` folders of cudnn to `bin`, `include`, and `lib` folders of `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\Vx.y` (x.y in Vx.y indicates cuda version).  

If TensorRT is used for inference acceleration under CUDA, TensorRT needs to be prepared, please refer to [Official Tutorial](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-zip).  
Copy the `.dll` file of the installation directory `lib` folder to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\Vx.y\bin`.

### 2.3 Prepare Paddle Inference C++ prediction library
Paddle Inference C++ prediction library provides different pre-compiled versions for different CPU and CUDA versions. You can choose the appropriate pre-compiled library according to your environment: [C++ prediction library download](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html#windows).

If the precompiled libraries provided do not meet the requirements, you can compile the Paddle Inference C++ prediction library by yourself, please refer to [Compile Tutorial](https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/source_compile.html).

This document takes CUDA=11.6, CUDNN=8.4.1.5, TensorRT=8.4.1.5 as an example to introduce.

Paddle Inference directory structure: 
```shell
D:\projects\paddle_inference
  ├── paddle
  ├── third_party
  ├── CMakeCache.txt
  └── version.txt
```

### 2.4 Prepare OpenCV
This example uses OpenCV to read pictures, so you need to install OpenCV. In other projects, you can install as needed.

1. Download opencv-4.6.0 for Windows platform, [Download link](https://sourceforge.net/projects/opencvlibrary/files/4.6.0/opencv-4.6.0-vc14_vc15.exe/download).  
2. Run the downloaded executable file and extract OpenCV to the specified directory, such as `D:\projects\opencv`.
3. Configure the environment variables, as shown in the following process (if you use the global absolute path, you can not set the environment variables)  
    - `My Computer`->`Propertie`->`Advanced System Settings`->`Environment Variables`
    - Find `Path` in the system variable (if not, create it by yourself), and double-click to edit.
    - Fill in the opencv path, such as`D:\projects\opencv\build\x64\vc15\bin`.


## 3. Prepare model and picture

You can download the prepared [inference model](https://paddleseg.bj.bcebos.com/dygraph/demo/pp_liteseg_infer_model.tar.gz) to the local for subsequent testing.
If you need to test other models, please refer to the [document](../../model_export.md) to export the inference model.

The inference model file format is as follows: 
```shell
pp_liteseg_infer_model
  ├── deploy.yaml            # Deployment related configuration file, mainly describing how data is preprocessed, etc.
  ├── model.pdmodel          # Topology file of inference model.
  ├── model.pdiparams        # Weight file of inference model.
  └── model.pdiparams.info   # Additional information of parameters.
```

`model.pdmodel` can be visualized by [Netron](https://netron.app/), click the input node to see the number of inputs and outputs and data types of the inference model (such as int32_t, int64_t, float, etc.).
If the output data type of the inference model is not int32_t, an error will be reported after executing the default code. At this time, you need to manually modify codes as the corresponded output data type in `deploy/cpp/src/test_seg.cc` as follows: 
```
std::vector<int32_t> out_data(out_num);
```

Download an [image](https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png) from the cityscapes validation set to the local for subsequent testing. 

## 4. Compile

The overall directory structure of the project is as follows:
```shell
D:\projects
  ├── opencv
  ├── paddle_inference
  └── PaddleSeg
```


### 4.1 Use CMake to generate project files

The description of compilation parameters is as follows, where `*` indicates that it is only specified when using **GPU version** prediction library, and `#` indicates that it is only specified when using **TensorRT**. 

| Parameters       | Description                                                                                                   |
|------------------|---------------------------------------------------------------------------------------------------------------|
| *WITH_GPU        | Whether to use GPU, the default is OFF;                                                                       |
| *CUDA_LIB        | Library path of CUDA;                                                                                         |
| *USE_TENSORRT    | Whether to use TensorRT, the default is OFF;                                                                  |
| #TENSORRT_DLL    | The .dll files storage path of TensorRT;                                                                      |
| WITH_MKL         | Whether to use MKL, the default is ON, which means to use MKL. If it is set to OFF, it means to use Openblas; |
| CMAKE_BUILD_TYPE | Specify to use Release or Debug when compiling;                                                               |
| PADDLE_LIB_NAME  | Paddlec Inference prediction library name;                                                                    |
| OPENCV_DIR       | The installation path of OpenCV;                                                                              |
| PADDLE_LIB       | The installation path of Paddle Inference prediction library;                                                 |
| DEMO_NAME        | Executable file name;                                                                                         |

Enter the `cpp` directory: 
```
cd D:\projects\PaddleSeg\deploy\cpp
```

Create the `build` folder and enter its directory: 
```commandline
mkdir build
cd build
```

The compilation command is executed in the following format:

(**Note**: If the path contains spaces, enclosed in quotes.)
```
cmake .. -G "Visual Studio 16 2019" -A x64 -T host=x64 -DUSE_TENSORRT=ON -DWITH_GPU=ON -DWITH_MKL=ON -DCMAKE_BUILD_TYPE=Release -DPADDLE_LIB_NAME=paddle_inference -DCUDA_LIB=path_to_cuda_lib -DOPENCV_DIR=path_to_opencv -DPADDLE_LIB=path_to_paddle_dir -DTENSORRT_DLL=path_to_tensorrt_.dll -DDEMO_NAME=test_seg
```

For example, GPU does not use TensorRT inference and the command is as follows:
```
cmake .. -G "Visual Studio 16 2019" -A x64 -T host=x64 -DUSE_TENSORRT=OFF -DWITH_GPU=ON -DWITH_MKL=ON -DCMAKE_BUILD_TYPE=Release -DPADDLE_LIB_NAME=paddle_inference -DCUDA_LIB="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\lib\x64" -DOPENCV_DIR=D:\projects\opencv -DPADDLE_LIB=D:\projects\paddle_inference -DDEMO_NAME=test_seg
```

GPU uses TensorRT inference, and the command is as follows:
```
cmake .. -G "Visual Studio 16 2019" -A x64 -T host=x64 -DUSE_TENSORRT=ON -DWITH_GPU=ON -DWITH_MKL=ON -DCMAKE_BUILD_TYPE=Release -DPADDLE_LIB_NAME=paddle_inference -DCUDA_LIB="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\lib\x64" -DOPENCV_DIR=D:\projects\opencv -DPADDLE_LIB=D:\projects\paddle_inference -DTENSORRT_DLL="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin" -DDEMO_NAME=test_seg
```

CPU uses MKL inference, and the command is as follows:
```
cmake .. -G "Visual Studio 16 2019" -A x64 -T host=x64 -DWITH_GPU=OFF -DWITH_MKL=ON -DCMAKE_BUILD_TYPE=Release -DPADDLE_LIB_NAME=paddle_inference -DOPENCV_DIR=D:\projects\opencv -DPADDLE_LIB=D:\projects\paddle_inference -DDEMO_NAME=test_seg
```

The CPU uses OpenBlas inference, and the command is as follows:
```
cmake .. -G "Visual Studio 16 2019" -A x64 -T host=x64 -DWITH_GPU=OFF -DWITH_MKL=OFF -DCMAKE_BUILD_TYPE=Release -DPADDLE_LIB_NAME=paddle_inference -DOPENCV_DIR=D:\projects\opencv -DPADDLE_LIB=D:\projects\paddle_inference -DDEMO_NAME=test_seg
```

### 4.2 Compile

Open `cpp\build\cpp_inference_demo.sln` with `Visual Studio 2019`, set the compilation mode to `Release`, click `Generate`->`Generate Solution`, and generate `test_seg.exe` in `cpp\build\Release`.

## 5、Execute

Enter the `build\Release` directory and put the prepared model and image into `test_seg.exe` peer directory, `build\Release` has the following structure:
```shell
Release
├──test_seg.exe                # Executable file.
├──cityscapes_demo.png         # Test picture.
├──pp_liteseg_infer_model      # Model used for inference.
    ├── deploy.yaml            # Deployment related configuration file, mainly describing how data is preprocessed, etc.
    ├── model.pdmodel          # Topology file of inference model.
    ├── model.pdiparams        # Weight file of inference model.
    └── model.pdiparams.info   # Additional information of parameters.
├──*.dll                       # dll files.
```

Run the following command for inference, GPU inference:
```commandline
test_seg.exe --model_dir=./pp_liteseg_infer_model --img_path=./cityscapes_demo.png --devices=GPU
```

CPU inference：
```commandline
test_seg.exe --model_dir=./pp_liteseg_infer_model --img_path=./cityscapes_demo.png --devices=CPU
```

Save predicted result as `out_img.jpg`, this image uses histogram equalization to facilitate visualization, as shown below: 

![out_img](https://user-images.githubusercontent.com/52520497/131456277-260352b5-4047-46d5-a38f-c50bbcfb6fd0.jpg)

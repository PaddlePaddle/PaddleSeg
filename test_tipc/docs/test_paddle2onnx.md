# Paddle2ONNX 测试

Paddle2ONNX 测试的主程序为`test_paddle2onnx.sh`，可以测试基于Paddle2ONNX的模型转换和onnx预测功能。


## 1. 测试结论汇总

- 推理相关：

| 算法名称 | 模型名称 | device_CPU | device_GPU | batchsize |
|  :----:   |  :----: |   :----:   |  :----:  |   :----:   |
|  PP_LiteSeg   |  pp_liteseg_stdc1 |  支持 | 支持 | 1 |
|  PP_LiteSeg   |  pp_liteseg_stdc2 |  支持 | 支持 | 1 |
|  ConnectNet   |  pp_humanseg_lite |  支持 | 支持 | 1 |
|  HRNet W18 Small   | pp_humanseg_mobile  |  支持 | 支持 | 1 |
|  DeepLabV3P ResNet50    |  pp_humanseg_server |  支持 | 支持 | 1 |
|  HRNet   |  fcn_hrnet_w18 |  支持 | 支持 | 1 |
|  OCRNet   |  ocrnet_hrnetw18 |  支持 | 支持 | 1 |
|  OCRNet   |  ocrnet_hrnetw48 |  支持 | 支持 | 1 |


## 2. 测试流程


### 2.1 准备环境


- 安装PaddlePaddle：如果您已经安装了2.2或者以上版本的paddlepaddle，那么无需运行下面的命令安装paddlepaddle。
    ```
    # 需要安装2.2及以上版本的Paddle
    # 安装GPU版本的Paddle
    pip install paddlepaddle-gpu==2.2.2
    # 安装CPU版本的Paddle
    pip install paddlepaddle==2.2.2
    ```

- 安装依赖
    ```
    pip install  -r requirements.txt
    ```

- 安装 Paddle2ONNX（自动安装）
    ```
    pip install paddle2onnx
    ```

- 安装 ONNXRuntime（自动安装）
    ```
    # 建议安装 1.9.0 版本，可根据环境更换版本号
    pip install onnxruntime==1.9.0
    ```


### 2.3 功能测试

测试方法如下所示，希望测试不同的模型文件，只需更换为自己的参数配置文件，即可完成对应模型的测试。

```bash
bash test_tipc/test_paddle2onnx.sh ${your_params_file} paddle2onnx_infer
```

以`pp_liteseg_stdc1`的`Paddle2ONNX 测试`为例，命令如下所示。

 ```bash
bash test_tipc/prepare.sh test_tipc/configs/pp_liteseg_stdc1/pp_liteseg_stdc1_model_linux_gpu_normal_normal_paddle2onnx_python_linux_cpu.txt paddle2onnx_infer
```

```bash
bash test_tipc/test_paddle2onnx.sh test_tipc/configs/pp_liteseg_stdc1/pp_liteseg_stdc1_model_linux_gpu_normal_normal_paddle2onnx_python_linux_cpu.txt paddle2onnx_infer
```

输出结果如下，表示命令运行成功。

```
 Run successfully with command - pp_liteseg_stdc1 -  paddle2onnx --model_dir=./test_tipc/infer_models/pp_liteseg_stdc1_fix_shape/ --model_filename=model.pdmodel --params_filename=model.pdiparams --save_file=./test_tipc/infer_models/pp_liteseg_stdc1_fix_shape/model.onnx --opset_version=11 --enable_onnx_checker=True!  
 Run successfully with command - pp_liteseg_stdc1 - python3.7 deploy/python/infer_onnx.py --img_path=test_tipc/cpp/cityscapes_demo.png --onnx_file=./test_tipc/infer_models/pp_liteseg_stdc1_fix_shape/model.onnx > ./log/pp_liteseg_stdc1//paddle2onnx_infer_cpu.log 2>&1 !
```

预测结果会自动保存在 `./log/pp_liteseg_stdc1/paddle2onnx_infer_cpu.log` ，可以看到onnx运行结果：
```
Predicted image is saved in ./output/cityscapes_demo.png
```

如果运行失败，也会在终端中输出运行失败的日志信息以及对应的运行命令。可以基于该命令，分析运行失败的原因。

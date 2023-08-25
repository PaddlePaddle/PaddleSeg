# TensorRT 安装说明

TensorRT是用于在NVDIA GPU上加速预测的C++库，可以搭配Paddle Inference使用，TensorRT安装步骤简单，但是经常容易安装不上，因此记录下安装过程。

## 1.【安装成功与否的关键】查阅适合自己cuda版本的TensorRT库:

这一步容易忽略，但是十分重要。安装不正确的包会导致出现无法排查的报错。

    > CUDA 工具包 10.2 配合 cuDNN v7.6.5, 如需使用 PaddleTensorRT 推理，需配合 TensorRT7.0.0.11 （经过验证，cuda10.2也适配TensorRT-7.1.3.4 和 TensorRT-7.2.3.4 )

    > CUDA 工具包 11.2 配合 cuDNN v8.2.1, 如需使用 PaddleTensorRT 推理，需配合 TensorRT8.0.3.4

    > CUDA 工具包 11.6 配合 cuDNN v8.4.0, 如需使用 PaddleTensorRT 推理，需配合 TensorRT8.4.0.6

    > CUDA 工具包 11.7 配合 cuDNN v8.4.1, 如需使用 PaddleTensorRT 推理，需配合 TensorRT8.4.2.4

    > CUDA 工具包 11.8 配合 cuDNN v8.6.0, 如需使用 PaddleTensorRT 推理，需配合 TensorRT8.5.1.7

    > CUDA 工具包 12.0 配合 cuDNN v8.9.1, 如需使用 PaddleTensorRT 推理，需配合 TensorRT8.6.1.6

## 2. 前往 [NVIDIA 官网](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#downloading)下载适合自己的TensorRT包。

## 3. 按照[官方文档](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar)安装：
执行到第四步导出路径即可。

## 4. 进行库依赖验证，查看自己是否有库文件没有找到：
参考[这里](https://github.com/PaddlePaddle/Paddle/issues/29362#issuecomment-1369440978)，并对对应的库文件进行链接到对应环境地址（库文件一般存在于anoconda3/env/envname/lib下）:
- 这里也有一点经验，如果发现没有找到的库文件和自己当前环境不匹配，可能说明TRT版本安装错了，例如我的环境是cuda10.2搭配cudnn7.6.5，但是提示我cudnn8的库文件缺失。
- 即便如此，提示什么库缺失，还是可以下载对应的库解压（比如我这里的cudnn8），并链接到对应的库排除掉找不到库依赖的问题，我后续就是链接到libcudnn8.so后正确推理的。

    ```bash
    # check the dependencies of libnvinfer.so.8
    ldd libnvinfer.so.8
    # output means all the dependencies can be found
            linux-vdso.so.1 =>  (0x00007ffcb671b000)
            libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f5f6c6a3000)
            libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f5f6c49f000)
            librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007f5f6c297000)
            libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f5f6bf15000)
            libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f5f6bc0c000)
            libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f5f6b9f6000)
            libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f5f6b62c000)
            /lib64/ld-linux-x86-64.so.2 (0x00007f5f8a0f7000)

    # check the dependencies of libnvinfer_plugin.so
    ldd libnvinfer_plugin.so

    # output means libcublas.so.11,   libcublasLt.so.11 and  libcudnn.so.8  can't be found
            linux-vdso.so.1 =>  (0x00007ffd566d7000)
            libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f70516c4000)
            libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f70514c0000)
            librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007f70512b8000)
            libcublas.so.11 => not found
            libcublasLt.so.11 => not found
            libcudnn.so.8 => not found
            libnvinfer.so.8 => /mnt/whs/envs/TensorRT-8.5.1.7/lib/libnvinfer.so.8 (0x00007f7033a81000)
            libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f70336ff000)
            libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f70333f6000)
            libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f70331e0000)
            libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f7032e16000)
            /lib64/ld-linux-x86-64.so.2 (0x00007f705424a000)

    # So find installed cublas&cudnn and add their path into LD_LIBRARY_PATH.
    And use ldd libnvinfer_plugin.so to recheck it.
    ```

## 5. 验证安装成功：
可以使用deploy/slim/act/test_seg.py进行验证。

    ```bash
    cd PaddleSeg/deploy/slim/act/

    wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

    python test_seg.py      \
            --model_path=liteseg_tiny_scale1.0   \  
            --dataset='cityscape'       \
            --image_file=cityscapes_demo.png       \
            --use_trt=True       \
            --precision=fp32       \
            --save_file res_qat_fp32.png
    ```


<td>
<img src="https://github.com/PaddlePaddle/PaddleSeg/assets/34859558/9a5adb32-190e-4b05-a52d-f6fdc8f56a29" width="2000" height="340">
</td>

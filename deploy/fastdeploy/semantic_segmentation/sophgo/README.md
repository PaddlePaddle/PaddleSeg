[English](README.md) | 简体中文
# PaddleSeg在算能（Sophgo）硬件上通过FastDeploy部署模型

## 1. 说明  
PaddleSeg支持部署的支持如下型号的Sophgo芯片的部署  
- Sophgo 1684X

PaddleSeg支持通过FastDeploy在算能TPU上部署相关Segmentation模型

## 2. 预导出的推理模型

为了方便开发者的测试，下面提供了PaddleSeg导出的部分推理模型，开发者可直接下载使用。

PaddleSeg训练模型导出为推理模型，请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/docs/model_export_cn.md)  

| 模型                              | 参数文件大小    |输入Shape |  mIoU | mIoU (flip) | mIoU (ms+flip) |
|:---------------------------------------------------------------- |:----- |:----- | :----- | :----- | :----- |
| [PP-LiteSeg-T(STDC1)-cityscapes-without-argmax](https://bj.bcebos.com/fastdeploy/models/rk1/ppliteseg.tar.gz)| 31MB  | 1024x512 | 77.04% | 77.73% | 77.46% |

## 3. 自行导出算能硬件支持的PaddleSeg模型
### 3.1 模型版本
支持PaddleSeg高于2.6版本的Segmentation模型，目前FastDeploy测试过可在算能TPU支持的模型如下：
- [PP-LiteSeg系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/configs/pp_liteseg/README.md)

### 3.2 将PaddleSeg推理模型转换为bmodel模型步骤

SOPHGO-TPU部署模型前需要将Paddle模型转换成bmodel模型，具体步骤如下:
- 下载Paddle模型[PP-LiteSeg-B(STDC2)-cityscapes-without-argmax](https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz)
- Paddle模型转换为ONNX模型，请参考[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)
- ONNX模型转换bmodel模型的过程，请参考[TPU-MLIR](https://github.com/sophgo/tpu-mlir)

### 3.3 bmode模型转换example

下面以[PP-LiteSeg-B(STDC2)-cityscapes-without-argmax](https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz)为例,教大家如何转换Paddle模型到SOPHGO-TPU支持的bmodel模型

- 下载PP-LiteSeg-B(STDC2)-cityscapes-without-argmax模型,并转换为ONNX模型
```shell
# 下载Paddle2ONNX仓库
git clone https://github.com/PaddlePaddle/Paddle2ONNX

# 下载Paddle静态图模型并为Paddle静态图模型固定输入shape
## 进入为Paddle静态图模型固定输入shape的目录
cd Paddle2ONNX/tools/paddle

wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz
tar xvf PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz

# 修改PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer模型的输入shape，由动态输入变成固定输入
python paddle_infer_shape.py --model_dir PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer \
                             --model_filename model.pdmodel \
                             --params_filename model.pdiparams \
                             --save_dir pp_liteseg_fix \
                             --input_shape_dict="{'x':[1,3,512,512]}"

#将固定输入的Paddle模型转换成ONNX模型
paddle2onnx --model_dir pp_liteseg_fix \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file pp_liteseg.onnx \
            --enable_dev_version True
```

- 导出bmodel模型

以转换BM1684x的bmodel模型为例子，我们需要下载[TPU-MLIR](https://github.com/sophgo/tpu-mlir)工程，安装过程具体参见[TPU-MLIR文档](https://github.com/sophgo/tpu-mlir/blob/master/README.md)。
#### 1.	安装
``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234是一个示例，也可以设置其他名字
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest

source ./envsetup.sh
./build.sh
```

#### 2.	ONNX模型转换为bmodel模型
``` shell
mkdir pp_liteseg && cd pp_liteseg

#在该文件中放入测试图片，同时将上一步转换的pp_liteseg.onnx放入该文件夹中
cp -rf ${REGRESSION_PATH}/dataset/COCO2017 .
cp -rf ${REGRESSION_PATH}/image .
#放入onnx模型文件pp_liteseg.onnx

mkdir workspace && cd workspace

#将ONNX模型转换为mlir模型，其中参数--output_names可以通过NETRON查看
model_transform.py \
    --model_name pp_liteseg \
    --model_def ../pp_liteseg.onnx \
    --input_shapes [[1,3,512,512]] \
    --mean 0.0,0.0,0.0 \
    --scale 0.0039216,0.0039216,0.0039216 \
    --keep_aspect_ratio \
    --pixel_format rgb \
    --output_names bilinear_interp_v2_6.tmp_0 \
    --test_input ../image/dog.jpg \
    --test_result pp_liteseg_top_outputs.npz \
    --mlir pp_liteseg.mlir

#将mlir模型转换为BM1684x的F32 bmodel模型
model_deploy.py \
  --mlir pp_liteseg.mlir \
  --quantize F32 \
  --chip bm1684x \
  --test_input pp_liteseg_in_f32.npz \
  --test_reference pp_liteseg_top_outputs.npz \
  --model pp_liteseg_1684x_f32.bmodel
```
最终获得可以在BM1684x上能够运行的bmodel模型pp_liteseg_1684x_f32.bmodel。如果需要进一步对模型进行加速，可以将ONNX模型转换为INT8 bmodel，具体步骤参见[TPU-MLIR文档](https://github.com/sophgo/tpu-mlir/blob/master/README.md)。

## 4. 详细的部署示例  
- [C++部署](./cpp)
- [Python部署](./python)

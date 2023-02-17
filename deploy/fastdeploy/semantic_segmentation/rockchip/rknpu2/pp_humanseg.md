[English](pp_humanseg_EN.md) | 简体中文
# PP-HumanSeg模型转换示例

## 转换模型
下面以Portait-PP-HumanSegV2_Lite(肖像分割模型)为例子，教大家如何转换PaddleSeg模型到RKNN模型。

```bash
# 下载Paddle2ONNX仓库
git clone https://github.com/PaddlePaddle/Paddle2ONNX

# 下载Paddle静态图模型并为Paddle静态图模型固定输入shape
## 进入为Paddle静态图模型固定输入shape的目录
cd Paddle2ONNX/tools/paddle
## 下载Paddle静态图模型并解压
wget https://bj.bcebos.com/paddlehub/fastdeploy/Portrait_PP_HumanSegV2_Lite_256x144_infer.tgz
tar xvf Portrait_PP_HumanSegV2_Lite_256x144_infer.tgz
python paddle_infer_shape.py --model_dir Portrait_PP_HumanSegV2_Lite_256x144_infer/ \
                             --model_filename model.pdmodel \
                             --params_filename model.pdiparams \
                             --save_dir Portrait_PP_HumanSegV2_Lite_256x144_infer \
                             --input_shape_dict="{'x':[1,3,144,256]}"

# 静态图转ONNX模型，注意，这里的save_file请和压缩包名对齐
paddle2onnx --model_dir Portrait_PP_HumanSegV2_Lite_256x144_infer \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file Portrait_PP_HumanSegV2_Lite_256x144_infer/Portrait_PP_HumanSegV2_Lite_256x144_infer.onnx \
            --enable_dev_version True

# ONNX模型转RKNN模型
# 将ONNX模型目录拷贝到Fastdeploy根目录
cp -r ./Portrait_PP_HumanSegV2_Lite_256x144_infer /path/to/Fastdeploy
# 转换模型,模型将生成在Portrait_PP_HumanSegV2_Lite_256x144_infer目录下
python tools/rknpu2/export.py \
        --config_path tools/rknpu2/config/Portrait_PP_HumanSegV2_Lite_256x144_infer.yaml \
        --target_platform rk3588
```

## 修改yaml配置文件

在**模型转换example**中，我们对模型的shape进行了固定，因此对应的yaml文件也要进行修改，如下:

**原yaml文件**
```yaml
Deploy:
  input_shape:
  - -1
  - 3
  - -1
  - -1
  model: model.pdmodel
  output_dtype: float32
  output_op: none
  params: model.pdiparams
  transforms:
  - target_size:
    - 256
    - 144
    type: Resize
  - type: Normalize
```

**修改后的yaml文件**
```yaml
Deploy:
  input_shape:
  - 1
  - 3
  - 144
  - 256
  model: model.pdmodel
  output_dtype: float32
  output_op: none
  params: model.pdiparams
  transforms:
  - target_size:
    - 256
    - 144
    type: Resize
  - type: Normalize
```

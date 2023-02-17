English | [简体中文](pp_humanseg.md)
# PPHumanSeg Model Deployment

## Converting Model
The following is an example of Portait-PP-HumanSegV2_Lite (portrait segmentation model), showing how to convert PPSeg model to RKNN model.

```bash
# Download Paddle2ONNX repository.
git clone https://github.com/PaddlePaddle/Paddle2ONNX

# Download the Paddle static map model and fix the input shape.
## Go to the directory where the input shape is fixed for the Paddle static map model.
cd Paddle2ONNX/tools/paddle
## Download and unzip Paddle static map model.
wget https://bj.bcebos.com/paddlehub/fastdeploy/Portrait_PP_HumanSegV2_Lite_256x144_infer.tgz
tar xvf Portrait_PP_HumanSegV2_Lite_256x144_infer.tgz
python paddle_infer_shape.py --model_dir Portrait_PP_HumanSegV2_Lite_256x144_infer/ \
                             --model_filename model.pdmodel \
                             --params_filename model.pdiparams \
                             --save_dir Portrait_PP_HumanSegV2_Lite_256x144_infer \
                             --input_shape_dict="{'x':[1,3,144,256]}"

# Converting static map model to ONNX model, note that the save_file here aligns with the zip name.
paddle2onnx --model_dir Portrait_PP_HumanSegV2_Lite_256x144_infer \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file Portrait_PP_HumanSegV2_Lite_256x144_infer/Portrait_PP_HumanSegV2_Lite_256x144_infer.onnx \
            --enable_dev_version True

# Convert ONNX model to RKNN model.
# Copy the ONNX model directory to the Fastdeploy root directory.
cp -r ./Portrait_PP_HumanSegV2_Lite_256x144_infer /path/to/Fastdeploy
# Convert model, the model will be generated in the Portrait_PP_HumanSegV2_Lite_256x144_infer directory.
python tools/rknpu2/export.py \
        --config_path tools/rknpu2/config/Portrait_PP_HumanSegV2_Lite_256x144_infer.yaml \
        --target_platform rk3588
```

## Modify yaml Configuration File

In the **An example of Model Conversion** part, we fixed the shape of the model, so the corresponding yaml file needs to be modified as follows:

**The original yaml file**
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

**The modified yaml file**
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

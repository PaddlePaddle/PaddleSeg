English | [简体中文](model_export_cn.md)

# Model Export

The trained model needs to be exported as a prediction model before deployment.

This tutorial will show how to export a trained model。


## Acquire the Pre-training Model

In this example，BiseNetV2 model will be used. Run the following command or click [link](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/bisenet_cityscapes_1024x1024_160k/model.pdparams) to download the pretrained model.
```shell
mkdir bisenet && cd bisenet
wget https://paddleseg.bj.bcebos.com/dygraph/cityscapes/bisenet_cityscapes_1024x1024_160k/model.pdparams
cd ..
```

## Export the prediction Model

Make sure you have installed PaddleSeg and are in the PaddleSeg directory.

Run the following command, and the prediction model will be saved in `output` directory.

```shell
export CUDA_VISIBLE_DEVICES=0 # Set a usable GPU.
# If on windows, Run the following command：
# set CUDA_VISIBLE_DEVICES=0
python export.py \
       --config configs/bisenet/bisenet_cityscapes_1024x1024_160k.yml \
       --model_path bisenet/model.pdparams\
       --save_dir output
```

### Description of Exported Script Parameters

|parammeter|purpose|is needed|default|
|-|-|-|-|
|config|Config file|yes|-|
|save_dir|Save root path for model and VisualDL log files|no|output|
|model_path|Path of pre-training model parameters|no|The value in config file|
|with_softmax|Add softmax operator at the end of the network. Since PaddleSeg networking returns Logits by default, you can set it to True if you want the deployment model to get the probability value|no|False|
|without_argmax|Whether or not to add argmax operator at the end of the network. Since PaddleSeg networking returns Logits by default, we add argmax operator at the end of the network by default in order to directly obtain the prediction results for the deployment model|no|False|
|input_shape| Set the input shape of exported model, such as `--input_shape 1 3 1024 1024`。if input_shape is not provided，the Default input shape of exported model is [-1, 3, -1, -1] | no | None |

## Prediction Model Files

```shell
output
  ├── deploy.yaml            # Config file of deployment
  ├── model.pdiparams        # Paramters of static model
  ├── model.pdiparams.info   # Additional information witch is not concerned generally
  └── model.pdmodel          # Static model file
```

After exporting prediction model, it can be deployed by the following methods.

|Deployment scenarios|Inference library|Tutorial|
|-|-|-|
|Server (Nvidia GPU and X86 CPU) Python deployment|Paddle Inference|[doc](../deploy/python/)|
|Server (Nvidia GPU and X86 CPU) C++ deployment|Paddle Inference|[doc](../deploy/cpp/)|
|Mobile deployment|Paddle Lite|[doc](../deploy/lite/)|
|Service-oriented deployment |Paddle Serving|[doc](../deploy/serving/)|
|Web deployment|Paddle JS|[doc](../deploy/web/)|

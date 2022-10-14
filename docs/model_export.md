English | [简体中文](model_export_cn.md)

# Model Export

After model training, we can export the inference model and deploy it using inference library, which achieves faster inference speed.

This tutorial will show how to export a trained model。


## Acquire trained weight

After model training, the weight with the highest accuracy is saved in ` path/to/save/best_ model/model.pdparams`.

For the convenience of this demo, we run the following commands to download the [trained weight](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k/model.pdparams) of PP-LiteSeg.


```shell
wget https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k/model.pdparams
```

## Export the prediction Model

Run the following command in the root of PaddleSeg, the inference model is saved in `output/inference_model`.

```shell
python tools/export.py \
       --config configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k.yml \
       --model_path model.pdparams \
       --save_dir output/inference_model
```

**Description of Exported Script Parameters**

|Parammeter|Purpose|Is Needed|Default|
|-|-|-|-|
|config|The path of config file|yes|-|
|model_path|The path of trained weight|no|-|
|save_dir| The save dir for the inference model|no|`output/inference_model`|
|input_shape| Set the input shape (`N*C*H*W`) of the inference model, such as `--input_shape 1 3 1024 1024`。if input_shape is not provided，the input shape of the inference model is [-1, 3, -1, -1]. If the image shape in prediction is fixed, you should set the input_shape. | no  | None |
|output_op | Set the op that is appended to the inference model, should in [`argmax`, `softmax`, `none`]. PaddleSeg models outputs logits (`N*C*H*W`) by default. Adding `argmax` operation, we get the label for every pixel, the dimension of output is `N*H*W`. Adding `softmax` operation, we get the probability of different classes for every pixel. | no | argmax |
|with_softmax| Deprecated params, please use --output_op. Add softmax operator at the end of the network. Since PaddleSeg networking returns Logits by default, you can set it to True if you want the deployment model to get the probability value|no|False|
|without_argmax|Deprecated params, please use --output_op. Whether or not to add argmax operator at the end of the network. Since PaddleSeg networking returns Logits by default, we add argmax operator at the end of the network by default in order to directly obtain the prediction results for the deployment model|no|False|


Note that:
* If you encounter shape-relevant issue, please try to set the input_shape.

## Prediction Model Files

```shell
output/inference_model
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

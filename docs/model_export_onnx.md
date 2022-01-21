English|[简体中文](model_export_onnx_cn.md)
# Export model with ONNX format

After training the model by PaddleSeg, we also support exporting model with ONNX format. This tutorial provides an example to introduce it.

For the complete method of exporting ONNX format models, please refer to [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)。

## 1.Export the inference model

Refer to [document](./model_export.md) to export model, and save the exported inference model to the output folder, as follows.


```shell
./output
  ├── deploy.yaml            # deployment-related profile
  ├── model.pdmodel          # topology file of inference model
  ├── model.pdiparams        # weight file of inference model
  └── model.pdiparams.info   # additional information, generally do not need attention to this file
```

## 2. Export ONNX format model

Install Paddle2ONNX (version 0.6 or higher).

```
pip install paddle2onnx
```

Execute the following command to export the prediction model in the output folder to an ONNX format model by Paddle2ONNX.
```
paddle2onnx --model_dir output \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --opset_version 11 \
            --save_file output.onnx
```

The exported ONNX format model is saved as output.onnx file.

Reference documents:
* [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)
* [ONNX](https://onnx.ai/)



# UAPI for PaddleSeg: Quick Start

## 1 Install Dependencies

### 1.1 Install PaddlePaddle

Please follow the instruction on [PaddlePaddle official website](https://www.paddlepaddle.org.cn/).

### 1.2 Install PaddleSeg

Please follow [the installation docs of PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/docs/install.md). If you plan to develop for PaddleSeg on UAPI, we strongly recommend [installing PaddleSeg from source](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/docs/install.md#221-install-paddleseg-from-source).


## 2 Experience UAPI Through Demo

### 2.1 Prepare Dataset for Testing

Create a directory named `uapi_demo` in the root directory of PaddleSeg repo. After that, download the demo dataset from [here](https://paddleseg.bj.bcebos.com/humanseg/data/mini_supervisely.zip). Unzip the files to `uapi_demo/data/mini_supervisely`.

### 2.2 Run Demo Script

Switch to the root directory of PaddleSeg repo if you are not there. Then run the following commands:

```shell
python -m uapi.demo
```

Check out the training output files in `uapi_demo/output/`, the prediction (with a dynamic-graph model) results in `uapi_demo/output/pred_res`, the exported model in `uapi_demo/output/infer`, the inference (with a static-graph model) results in `uapi_demo/output/infer_res`, and the compression (quantization aware training and export) outputs in `uapi_demo/output/infer_res/compress`.

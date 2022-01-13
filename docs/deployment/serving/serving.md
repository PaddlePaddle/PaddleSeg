English|[简体中文](serving_cn.md)
# Paddle Serving deployment

## Overview

The model trained by PaddleSeg can be deployed as a service using [Paddle Serving](https://github.com/PaddlePaddle/Serving).

This turtorial introduces the deployment method using Paddle Serving. For more details, please refer to the [document](https://github.com/PaddlePaddle/Serving/blob/v0.6.0/README_CN.md).


## Environmental preparation

Environment preparations are required on the server side and the client side. Please refer to [document](https://github.com/PaddlePaddle/Serving/blob/v0.6.0/README_CN.md#%E5%AE%89%E8%A3%85) for more details.

On the server side:
* Install PaddlePaddle (version>=2.0)
* Install paddle-serving-app (version>=0.6.0)
* Install paddle-serving-server or paddle-serving-server-gpu (version>=0.6.0)

    ```shell
    pip3 install paddle-serving-app==0.6.0

    # CPU
    pip3 install paddle-serving-server==0.6.0

    # Choose paddle-serving-server-gpu according to your GPU environment
    pip3 install paddle-serving-server-gpu==0.6.0.post102 #GPU with CUDA10.2 + TensorRT7
    pip3 install paddle-serving-server-gpu==0.6.0.post101 # GPU with CUDA10.1 + TensorRT6
    pip3 install paddle-serving-server-gpu==0.6.0.post11 # GPU with CUDA10.1 + TensorRT7
    ```

On the client side:
* Install paddle-serving-app (version>=0.6.0)
* Install paddle-serving-client (version>=0.6.0)

    ```shell
    pip3 install paddle-serving-app==0.6.0
    pip3 install paddle-serving-client==0.6.0
    ```
 
## Prepare model and data

Download the [sample model](https://paddleseg.bj.bcebos.com/dygraph/demo/bisenet_demo_model.tar.gz) for testing. If you want to use other models, please refer to [model export tool](../../model_export.md).

```shell
$ wget https://paddleseg.bj.bcebos.com/dygraph/demo/bisenet_demo_model.tar.gz
tar zxvf bisenet_demo_model.tar.gz
```

Download a [picture](https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png) from cityscape to test.  If your model is trained on other datasets, please prepare test images by yourself.

```shell
$ wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png
```

## Convert model

Before Paddle Serving is deployed, we need to convert the prediction model. For details, please refer to the [document](https://github.com/PaddlePaddle/Serving/blob/v0.6.0/doc/SAVE_CN.md).

On the client side, execute the following script to convert the sample model.

```shell
python -m paddle_serving_client.convert \
    --dirname ./bisenetv2_demo_model \
    --model_filename model.pdmodel \
    --params_filename model.pdiparams
```

After excuting the script, the "serving_server" folder in the current directory saves the server model and configuration, and the "serving_client" folder saves the client model and configuration.

## Server Deployment

You can use `paddle_serving_server.serve` to start the RPC service, please refer to the [document](https://github.com/PaddlePaddle/Serving/blob/v0.6.0/README_CN.md#rpc%E6%9C%8D%E5%8A%A1).

If you finish to prepare environment on server side, export the  server model and  serving_server file, execute the following command to start the service. We use port 9292 on the server side. The server ip can be inquired by `hostname -i`.

```shell
python -m paddle_serving_server.serve \
    --model serving_server \
    --thread 10 \
    --port 9292 \
    --ir_optim
```

## Client request service

```
cd PaddleSeg/deploy/serving
```

Set the path of the serving_client file, the server-side ip and port, and the path of the test picture, and execute the following commands.

```shell
python test_serving.py \
    --serving_client_path path/to/serving_client \
    --serving_ip_port ip:port \
    --image_path path/to/image\
```

After the execution is complete, the divided image is saved in "result.png" in the current directory.

![cityscape_predict_demo.png](../../images/cityscapes_predict_demo.png)

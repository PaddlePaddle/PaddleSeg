English | [简体中文](README_CN.md)
# PaddleSeg Serving Deployment Demo

The PaddleSeg serving deployment Demo is built with FastDeploy Serving. FastDeploy Serving is a service-oriented deployment framework suitable for high-concurrency and high-throughput requests encapsulated based on the Triton Inference Server framework. It is a complete and high-performance service-oriented deployment framework that can be used in actual production. If you don’t need high-concurrency and high-throughput scenarios, and just want to quickly test the feasibility of online deployment of the model, please refer to [fastdeploy_serving](../simple_serving/)

## Environment

Before serving deployment, it is necessary to confirm the hardware and software environment requirements of the service image and the image pull command, please refer to [FastDeploy service deployment](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/README.md)

## Launch Serving

```bash
# Download demo code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/segmentation/paddleseg/serving

#Download PP_LiteSeg model file
wget  https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer.tgz
tar -xvf PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer.tgz

# Move the model files to models/infer/1
mv yolov5s.onnx models/infer/1/

# Pull fastdeploy image, x.y.z is FastDeploy version, example 1.0.2.
docker pull paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10

# Run the docker. The docker name is fd_serving, and the current directory is mounted as the docker's /serving directory
nvidia-docker run -it --net=host --name fd_serving -v `pwd`/:/serving paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10  bash

# Start the service (Without setting the CUDA_VISIBLE_DEVICES environment variable, it will have scheduling privileges for all GPU cards)
CUDA_VISIBLE_DEVICES=0 fastdeployserver --model-repository=/serving/models --backend-config=python,shm-default-byte-size=10485760
```

Output the following contents if serving is launched

```
......
I0928 04:51:15.784517 206 grpc_server.cc:4117] Started GRPCInferenceService at 0.0.0.0:8001
I0928 04:51:15.785177 206 http_server.cc:2815] Started HTTPService at 0.0.0.0:8000
I0928 04:51:15.826578 206 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```

## Client Requests

Execute the following command in the physical machine to send a grpc request and output the result

```
#Download test images
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

#Installing client-side dependencies
python3 -m pip install tritonclient\[all\]

# Send requests
python3 paddleseg_grpc_client.py
```

When the request is sent successfully, the results are returned in json format and printed out:

```

```

## Modify Configs



The default is to run ONNXRuntime on CPU. If developers need to run it on GPU or other inference engines, please see the  [Configs File](../../../../../serving/docs/EN/model_configuration-en.md) to modify the configs in `models/runtime/config.pbtxt`.

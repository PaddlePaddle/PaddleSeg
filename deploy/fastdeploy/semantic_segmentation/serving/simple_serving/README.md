English | [简体中文](README_CN.md)

# PaddleSeg Python Simple Serving Demo

PaddleSeg Python Simple serving is an example of serving deployment built by FastDeploy based on the Flask framework that can quickly verify the feasibility of online model deployment. It completes AI inference tasks based on http requests, and is suitable for simple scenarios without concurrent inference task. For high concurrency and high throughput scenarios, please refer to [fastdeploy_serving](../fastdeploy_serving/)

## 1. Environment

- 1. Prepare environment and install FastDeploy Python whl, refer to [download_prebuilt_libraries](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/en/build_and_install#install-prebuilt-fastdeploy)

## 2. Launch Serving
```bash
# Download demo code
git clone https://github.com/PaddlePaddle/PaddleSeg.git 
git checkout develop
cd PaddleSeg/deploy/fastdeploy/semantic_segmentation/serving/simple_serving

# Download PP_LiteSeg model
wget  https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer.tgz
tar -xvf PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer.tgz

# Launch server, change the configurations in server.py to select hardware, backend, etc.
# and use --host, --port to specify IP and port
fastdeploy simple_serving --app server:app
```

## 3. Client Requests
```bash
# Download test image
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

# Send request and get inference result (Please adapt the IP and port if necessary)
python client.py
```

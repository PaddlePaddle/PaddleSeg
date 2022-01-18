# sample deploy app using onnxruntime #
***

## Dependencies ##
***

- [onnxruntime](https://github.com/microsoft/onnxruntime)
  * tested with v1.10.0
  * to build onnxruntime from source you need cmake>=3.18
  * you can build from source with the [build script](https://github.com/microsoft/onnxruntime/blob/master/build.sh)
  * here is the sample procedure to build cpu onnxruntime
  ```bash
    readonly ONNXRUNTIME_VERSION="v1.10.0"
    git clone --recursive -b ${ONNXRUNTIME_VERSION} https://github.com/Microsoft/onnxruntime
    ./build.sh --config RelWithDebInfo --build_shared_lib --skip_tests --parallel `nproc`
    cd build/Linux/RelWithDebInfo
    make install
    ```

- opencv
```bash
sudo apt-get install libopencv-dev
```

## How to Build ##
***
```bash
mkdir build && cd build && cmake ../ && make -j`nproc`
```

## How to Run ##
***

- Download test images
```bash
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png
```

- Export PaddleSeg Model to onnx format
  * [export PaddleSeg model](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/docs/model_export.md)
  * convert exported model to onnx format with [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)

- Run app

```bash
./build/onnxruntime_cpp_inference_demo_app /path/to/test/image /path/to/onnx/model
```

# Dependencies #

- [onnxruntime](https://github.com/microsoft/onnxruntime)
  * tested with v1.10.0
  * you can build from source with the [build script](https://github.com/microsoft/onnxruntime/blob/master/build.sh)
  * you can also build docker environment for testing using dockerfiles from [here](https://github.com/microsoft/onnxruntime/tree/master/dockerfiles)

- opencv
```bash
sudo apt-get install libopencv-dev
```

# How to Run #

- Download test images

```bash
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png
```

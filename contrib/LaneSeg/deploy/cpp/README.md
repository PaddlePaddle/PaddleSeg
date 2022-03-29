English | [简体中文](README_CN.md)

## Deploy the PaddleSeg model using Paddle Inference C++


### 1、Install

- Paddle Inference C++

- OpenCV

- Yaml

 More install informations，please refer to [Tutorial](../../../../docs/deployment/inference/cpp_inference.md)。

### 2、Models and Pictures

 - Downdload model

   Enter to `LaneSeg/` directory, and execute commands as follows:
```shell
   mkdir output # if not exists
   wget -P output https://paddleseg.bj.bcebos.com/lane_seg/bisenet/model.pdparams
```
 - Export Model

```shell
   python export.py \
    --config configs/bisenetV2_tusimple_640x368_300k.yml \
    --model_path output/model.pdparams \
    --save_dir output/export
```  

 - Using the image `data/test_images/3.jpg`

### 3、Compile and execute

Enter to the `LaneSeg/deploy/cpp`

Execute `sh run_seg_cpu.sh`, it will compile and then perform prediction on X86 CPU.

Execute `sh run_seg_gpu.sh`, it will compile and then perform prediction on Nvidia GPU.

The result will be saved in the`out_img_seg.jpg` and `out_image_points.jpg` images

- Note：For the path of the model and image, you can change the files `run_seg_cpu.sh` and `run_seg_gpu.sh` as needed

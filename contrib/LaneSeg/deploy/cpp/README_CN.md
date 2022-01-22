简体中文 | [English](README.md)

## 使用Paddle Inference C++部署PaddleSeg模型

### 1、安装

- Paddle Inference C++

- OpenCV

- Yaml

 更多的安装信息，请参考[教程](../../../../docs/deployment/inference/cpp_inference_cn.md)。

### 2、模型和图片

 - 下载模型

 进入`LaneSeg/`目录下，执行如下命令:
```shell
   mkdir output # if not exists
   wget -P output https://paddleseg.bj.bcebos.com/lane_seg/bisenet/model.pdparams
```
 - 导出模型
```shell
   python export.py \
    --config configs/bisenetV2_tusimple_640x368_300k.yml \
    --model_path output/model.pdparams \
    --save_dir output/export
```  

 - 图片使用 `data/test_images/3.jpg`

### 3、编译、执行

进入目录`LaneSeg/deploy/cpp`

执行`sh run_seg_cpu.sh`，会进行编译，然后在X86 CPU上执行预测。

执行`sh run_seg_gpu.sh`，会进行编译，然后在Nvidia GPU上执行预测。

结果会保存在当前目录的`out_img_seg.jpg`和`out_image_points.jpg`图片。

- 注意：对于模型和图片的路径，可以按需要对文件`run_seg_cpu.sh`和`run_seg_gpu.sh`进行修改。

# PP-LiteSeg: A Superior Real-Time Semantic Segmentation Model

## Reference

> Juncai Peng, Yi Liu, Shiyu Tang, Yuying Hao, Lutao Chu, Guowei Chen, Zewu Wu, Zeyu Chen, Zhiliang Yu, Yuning Du, Qingqing Dang,Baohua Lai, Qiwen Liu, Xiaoguang Hu, Dianhai Yu, Yanjun Ma. PP-LiteSeg: A Superior Real-Time Semantic Segmentation Model. https://arxiv.org/abs/2204.02681

## Overview

We propose PP-LiteSeg, a novel lightweight model for the real-time semantic segmentation task. Specifically, we present a Flexible and Lightweight Decoder (FLD) to reduce computation overhead of previous decoder. To strengthen feature representations, we propose a Unified Attention Fusion Module (UAFM), which takes advantage of spatial and channel attention to produce a weight and then fuses the input features with the weight. Moreover, a Simple Pyramid Pooling Module (SPPM) is proposed to aggregate global context with low computation cost.

<div align="center">
<img src="https://user-images.githubusercontent.com/52520497/162148786-c8b91fd1-d006-4bad-8599-556daf959a75.png" width = "600" height = "300" alt="arch"  />
</div>


## Training

**Prepare:**
* Install gpu driver, cuda toolkit and cudnn
* Install Paddle and PaddleSeg ([doc](../../docs/install.md))
* Download dataset and link it to `PaddleSeg/data` ([Cityscapes](https://paddleseg.bj.bcebos.com/dataset/cityscapes.tar), [CamVid](https://paddleseg.bj.bcebos.com/dataset/camvid.tar))
    ```
    PaddleSeg/data
    ├── cityscapes
    │   ├── gtFine
    │   ├── infer.list
    │   ├── leftImg8bit
    │   ├── test.list
    │   ├── train.list
    │   ├── trainval.list
    │   └── val.list
    ├── camvid
    │   ├── annot
    │   ├── images
    │   ├── README.md
    │   ├── test.txt
    │   ├── train.txt
    │   └── val.txt
    ```

**Training:**

The config files of PP-LiteSeg are under `PaddleSeg/configs/pp_liteseg/`.

Based on the `train.py` script, we set the config file and start training model.

```Shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
export model=pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k     # test resolution is 1024*512
# export model=pp_liteseg_stdc1_cityscapes_1024x512_scale0.75_160k  # test resolution is 1536x768
# export model=pp_liteseg_stdc1_cityscapes_1024x512_scale1.0_160k   # test resolution is 2048x1024
# export model=pp_liteseg_stdc2_cityscapes_1024x512_scale0.5_160k
# export model=pp_liteseg_stdc2_cityscapes_1024x512_scale0.75_160k
# export model=pp_liteseg_stdc2_cityscapes_1024x512_scale1.0_160k
# export model=pp_liteseg_stdc1_camvid_960x720_10k
# export model=pp_liteseg_stdc2_camvid_960x720_10k
python -m paddle.distributed.launch tools/train.py \
    --config configs/pp_liteseg/${model}.yml \
    --save_dir output/${model} \
    --save_interval 1000 \
    --num_workers 3 \
    --do_eval \
    --use_vdl
```

After the training, the weights are saved in `PaddleSeg/output/xxx/best_model/model.pdparams`.

Refer to [doc](../../docs/train/train.md) for the detailed usage of training.

## Evaluation

With the config file and trained weights, we use the `val.py` script to evaluate the model.

Refer to [doc](../../docs/evaluation/evaluate/evaluate.md) for the detailed usage of evalution.

```shell
export CUDA_VISIBLE_DEVICES=0
export model=pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k
# export other model
python tools/val.py \
    --config configs/pp_liteseg/${model}.yml \
    --model_path output/${model}/best_model/model.pdparams \
    --num_workers 3
```

## Deployment

**Using ONNX+TRT**

Prepare:
* Install gpu driver, cuda toolkit and cudnn
* Download TensorRT 7 tar file from [Nvidia](https://developer.nvidia.com/tensorrt). We provide [cuda10.2-cudnn8.0-trt7.1](https://paddle-inference-dist.bj.bcebos.com/tensorrt_test/cuda10.2-cudnn8.0-trt7.1.tgz)
* Install the TensorRT whl in the tar file, i.e., `pip install TensorRT-7.1.3.4/python/xx.whl`
* Set Path, i.e., `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:TensorRT-7.1.3.4/lib`
* Install Paddle and PaddleSeg ([doc](../../docs/install.md))
* Run `pip install 'pycuda>=2019.1.1'`
* Run `pip install paddle2onnx onnx onnxruntime`


We measure the inference speed with [infer_onnx_trt.py](../../deploy/python/infer_onnx_trt.py), which first exports the Paddle model as ONNX and then infers the ONNX model by TRT.
Sometimes, the adaptive average pooling op can not be converted to ONNX. To solve the problem, you can adjust the input shape of the model as a multiple of 128.

```shell
python deploy/python/infer_onnx_trt.py \
    --config configs/pp_liteseg/pp_liteseg_xxx.yml
    --width 1024 \
    --height 512
```

Please refer to [infer_onnx_trt.py](../../deploy/python/infer_onnx_trt.py) for the detailed usage.

**Using PaddleInference**

Export the trained model as inference model ([doc](../../docs/model_export.md)).

Use PaddleInference to deploy the inference model on Nvidia GPU and X86 CPU([python api doc](../../docs/deployment/inference/python_inference.md), [cpp api doc](../../docs/deployment/inference/cpp_inference.md)).

## Performance

### Cityscapes

| Model | Backbone | Training Iters | Train Resolution | Test Resolution | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|-|
|PP-LiteSeg-T|STDC1|160000|1024x512|1025x512|73.10%|73.89%|-|[config](./pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k.yml)\|[model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k/model.pdparams)\|[log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k/train.log)\|[vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=66db3a2815980e41274ad587df2cd4e4)|
|PP-LiteSeg-T|STDC1|160000|1024x512|1536x768|76.03%|76.74%|-|[config](./pp_liteseg_stdc1_cityscapes_1024x512_scale0.75_160k.yml)\|[model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc1_cityscapes_1024x512_scale0.75_160k/model.pdparams)\|[log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc1_cityscapes_1024x512_scale0.75_160k/train.log)\|[vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=ea5d56fbfceb8d020eabe46e9bc8c40c)|
|PP-LiteSeg-T|STDC1|160000|1024x512|2048x1024|77.04%|77.73%|77.46%|[config](./pp_liteseg_stdc1_cityscapes_1024x512_scale1.0_160k.yml)\|[model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc1_cityscapes_1024x512_scale1.0_160k/model.pdparams)\|[log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc1_cityscapes_1024x512_scale1.0_160k/train.log)\|[vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=b9d2ca9445c5b3ee41db8ec37252d3e8)|
|PP-LiteSeg-B|STDC2|160000|1024x512|1024x512|75.25%|75.65%|-|[config](./pp_liteseg_stdc2_cityscapes_1024x512_scale0.5_160k.yml)\|[model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc2_cityscapes_1024x512_scale0.5_160k/model.pdparams)\|[log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc2_cityscapes_1024x512_scale0.5_160k/train.log)\|[vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=75a52ed995914223474b3c17e628d65e)|
|PP-LiteSeg-B|STDC2|160000|1024x512|1536x768|78.75%|79.23%|-|[config](./pp_liteseg_stdc2_cityscapes_1024x512_scale0.75_160k.yml)\|[model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc2_cityscapes_1024x512_scale0.75_160k/model.pdparams)\|[log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc2_cityscapes_1024x512_scale0.75_160k/train.log)\|[vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=a248fe1f645018306f1d4a0da33d97d6)|
|PP-LiteSeg-B|STDC2|160000|1024x512|2048x1024|79.04%|79.52%|79.85%|[config](./pp_liteseg_stdc2_cityscapes_1024x512_scale1.0_160k.yml)\|[model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc2_cityscapes_1024x512_scale1.0_160k/model.pdparams)\|[log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc2_cityscapes_1024x512_scale1.0_160k/train.log)\|[vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=12fa0144ca6a1541186afd2c53d31bcb)|

Note that:
* Use [infer_onnx_trt.py](../../deploy/python/infer_onnx_trt.py) to measure the inference speed.
* The flip denotes flip_horizontal, the ms denotes multi scale, i.e (0.75, 1.0, 1.25) * test_resolution.
* Simliar to other models in PaddleSeg, the mIoU in above table refer to the evaluation of PP-LiteSeg on Cityscapes validation set.
* You can download the trained model in above table and use it in evaluation.


**The comparisons with state-of-the-art real-time methods on Cityscapes as follows.**

<div align="center">

|Model|Encoder|Resolution|mIoU(Val)|mIoU(Test)|FPS|
|-|-|-|-|-|-|
ENet          | -           |  512x1024   | -    | 58.3 | 76.9  |
ICNet         | PSPNet50    |  1024x2048  | -    | 69.5 | 30.3  |
ESPNet        | ESPNet      |  512x1024   | -    | 60.3 | 112.9 |
ESPNetV2      | ESPNetV2    |  512x1024   | 66.4 | 66.2 | -     |
SwiftNet      | ResNet18    |  1024x2048  | 75.4 | 75.5 | 39.9  |
BiSeNetV1     | Xception39  |  768x1536   | 69.0 | 68.4 | 105.8 |
BiSeNetV1-L   | ResNet18    |  768x1536   | 74.8 | 74.7 | 65.5  |
BiSeNetV2     | -           |  512x1024   | 73.4 | 72.6 | 156   |
BiSeNetV2-L   | -           |  512x1024   | 75.8 | 75.3 | 47.3  |
FasterSeg     | -           |  1024x2048  | 73.1 | 71.5 | 163.9 |
SFNet         | DF1         |  1024x2048  | -    | 74.5 | 121   |
STDC1-Seg50  | STDC1       |  512x1024   | 72.2 | 71.9 | 250.4 |
STDC2-Seg50  | STDC2       |  512x1024   | 74.2 | 73.4 | 188.6 |
STDC1-Seg75  | STDC1       |  768x1536   | 74.5 | 75.3 | 126.7 |
STDC2-Seg75  | STDC2       |  768x1536   | 77.0 | 76.8 | 97.0 |
PP-LiteSeg-T1 | STDC1      |  512x1024  | 73.1 | 72.0 | 273.6  |
PP-LiteSeg-B1 | STDC2      |  512x1024  | 75.3 | 73.9 | 195.3 |
PP-LiteSeg-T2 | STDC1      |  768x1536  | 76.0 | 74.9 | 143.6 |
PP-LiteSeg-B2 | STDC2      |  768x1536  | 78.2 | 77.5 | 102.6|

</div>

<div align="center">
<img src="https://user-images.githubusercontent.com/52520497/162148733-70be896a-eadb-4790-94e5-f48dad356b2d.png" width = "500" height = "430" alt="iou_fps"  />
</div>

### CamVid

| Model | Backbone | Training Iters | Train Resolution | Test Resolution | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|-|
|PP-LiteSeg-T|STDC1|10000|960x720|960x720|73.30%|73.89%|73.66%|[config](./pp_liteseg_stdc1_camvid_960x720_10k.yml)\|[model](https://paddleseg.bj.bcebos.com/dygraph/camvid/pp_liteseg_stdc1_camvid_960x720_10k/model.pdparams)\|[log](https://paddleseg.bj.bcebos.com/dygraph/camvid/pp_liteseg_stdc1_camvid_960x720_10k/train.log)\|[vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=5685c196ff76493cecf867564c7e49be)|
|PP-LiteSeg-B|STDC2|10000|960x720|960x720|75.10%|75.85%|75.48%|[config](./pp_liteseg_stdc2_camvid_960x720_10k.yml)\|[model](https://paddleseg.bj.bcebos.com/dygraph/camvid/pp_liteseg_stdc2_camvid_960x720_10k/model.pdparams)\|[log](https://paddleseg.bj.bcebos.com/dygraph/camvid/pp_liteseg_stdc2_camvid_960x720_10k/train.log)\|[vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=cf5223dd121d58ceff7fd93135efb573)|

Note:
* The flip denotes flip_horizontal, the ms denotes multi scale, i.e (0.75, 1.0, 1.25) * test_resolution.
* The mIoU in above table refer to the evaluation of PP-LiteSeg on CamVid test set.

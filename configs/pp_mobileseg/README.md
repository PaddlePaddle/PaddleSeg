# PP-MobileSeg: Exploring Transformer Blocks for Efficient Mobile Segmentation.

## Reference
>

## Contents
1. Overview
2. Performance
3. Reproduction

## <img src="https://user-images.githubusercontent.com/34859558/190043857-bfbdaf8b-d2dc-4fff-81c7-e0aac50851f9.png" width="25"/> Overview
With the success of transformers in computer vision, several attempts have been made to adapt transformers to mobile devices. However, there still is room for optimization. Therefore, we propose PP-MobileSeg, a SOTA semantic segmentation model for mobile devices.

It is composed of three newly proposed parts, the strideformer backbone, the Aggregated Attention Module(AAM) and the Valid Interpolate Module(VIM):
* With the four-stage MobileNetV3 block as feature extractor, we manage to extract rich local features of different receptive field with little parameter overhead. Also, we further efficiently empower features from the last two stage with global view using strided sea attention.
* To effectively fuse the features, we use AAM to filter the detail features with ensemble voting and add the semantic feature to it to enhance the semantic information to the most content.
* At last, we use VIM to upsample the downsampled feature to the original resolution and significantly decrease latency. It only interpolates classes present in the final prediction which only takes around 10\% in the ADE20K dataset. This is a common scenario for datasets with large classes. Therefore it significantly decreases the latency of the final upsample process which takes the greatest part of the model's overall latency.


Extensive experiments show that PP-MobileSeg achieves a superior params-accuracy-latency tradeoff compared to other SOTA methods.
<div align="center">
<img src="https://user-images.githubusercontent.com/34859558/227450728-1338fcb1-3b8a-4453-a155-da60abcacb88.png"  width = "1000" />  
</div>



## <img src="https://user-images.githubusercontent.com/34859558/190044217-8f6befc2-7f20-473d-b356-148e06265205.png" width="25"/> Performance

### ADE20K
| Model | Backbone | Training Iters | Batchsize | Train Resolution | mIoU(%) | latency(ms)* | params(M) | Links |
|-|-|-|-|-|-|-|-|-|
|PP-MobileSeg-Base|StrideFormer-Base|80000|32|512x512|41.57%|265.5|5.62|[config](./pp_mobileseg_base_ade20k_512x512_160k.yml)\|[model](https://bj.bcebos.com/paddleseg/dygraph/ade20k/pp_mobileseg_base/model.pdparams)\|[log](https://bj.bcebos.com/paddleseg/dygraph/ade20k/pp_mobileseg_base/train.log)\|[vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=4836be3e2e571ec358a9cab069530fb2)\|[exported model](https://bj.bcebos.com/paddleseg/dygraph/ade20k/pp_mobileseg_base/export_model.zip)|
|PP-MobileSeg-Tiny|StrideFormer-Tiny|160000|16|512x512|36.70%|215.3|1.61|[config](./pp_mobileseg_tiny_ade20k_512x512_160k.yml)\|[model](https://bj.bcebos.com/paddleseg/dygraph/ade20k/pp_mobileseg_tiny/model.pdparams)\|[vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/index?id=8b48fc0ada781be47468bdeb6941eb99)\|[exported model](https://bj.bcebos.com/paddleseg/dygraph/ade20k/pp_mobileseg_tiny/export_model.zip)|


### Cityscapes
| Model | Backbone | Training Iters | Batchsize | Train Resolution | mIoU(%) | latency(ms)* | params(M) | Links |
|-|-|-|-|-|-|-|-|-|
|PP-MobileSeg-Base|StrideFormer-Base|80000|32|1024x512|77.83%|326.3|5.68|[config](./pp_mobileseg_base_cityscapes_1024x512_160k.yml)\|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/pp_mobileseg_base/model.pdparams)\|[log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/pp_mobileseg_base/train.log)\|[vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/index?id=735db0e47ddf783e6896f00bbb804819)\|[exported model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/pp_mobileseg_base/export_model.zip)|


### Compare with SOTA on ADE20Ks
| Model | Backbone | mIoU(%) | latency(ms)* | params(M) |
|-|-|-|-|-|
|LR-ASPP|MobileNetV3_large_x1_0|33.10|730.9|3.20|
|MobileSeg-Base|MobileNetV3_large_x1_0|33.26|391.5|2.85|
|TopFormer-Tiny|TopTransformer-Tiny|32.46|490.3|1.41|
|SeaFormer-Tiny|SeaFormer-Tiny|35.00|459.0|1.61|
|PP-MobileSeg-Tiny|StrideFormer-Tiny|36.70（**+1.7**）|215.3（**-48.2%**）|1.44(**-10.6%**)|
|TopFormer-Base|TopTransformer-Base|38.28|480.6|5.13|
|SeaFormer-Base**|SeaFormer-Base|40.07|465.4|8.64|
|PP-MobileSeg-Base|StrideFormer-Base|41.57(**+1.5**)|265.5(**-42.3%**)|5.62(**-34.9%**)|


### Ablation study of PP-MobileSeg-Base on ADE20K
| Model | Backbone | Train Resolution | mIoU(%) | latency(ms)* | params(M) | Links |
|-|-|-|-|-|-|-|
|baseline|Seaformer-Base|512x512|40.00%|465.6|8.27|[model](https://bj.bcebos.com/paddleseg/dygraph/ade20k/seaformer_base_ablation/model.pdprams)\|[log](https://bj.bcebos.com/paddleseg/dygraph/ade20k/seaformer_base_ablation/train.log)\|[vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=ac4847bef689ecd4e2c91d8e2674bfdb)\|[exported model](https://bj.bcebos.com/paddleseg/dygraph/ade20k/seaformer_base_ablation/export_model.zip)|
|+VIM|Seaformer-Base|512x512|40.07%|234.6|8.17|[model](https://bj.bcebos.com/paddleseg/dygraph/ade20k/seaformer_base_ablation/model.pdprams)\|[log](https://bj.bcebos.com/paddleseg/dygraph/ade20k/seaformer_base_ablation/train.log)\|[vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=ac4847bef689ecd4e2c91d8e2674bfdb)\|[exported model](https://bj.bcebos.com/paddleseg/dygraph/ade20k/seaformer_base_VIM_ablation/export_model.zip)|
|+VIM+StrideFormer|StrideFormer-Base|512x512|40.98%|235.1|5.54|[model](https://bj.bcebos.com/paddleseg/dygraph/ade20k/MV3_4stage_base_ablation/model.pdparams)\|[log](https://bj.bcebos.com/paddleseg/dygraph/ade20k/MV3_4stage_base_ablation/train.log)\|[vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=4836be3e2e571ec358a9cab069530fb2)\|[exported model](https://bj.bcebos.com/paddleseg/dygraph/ade20k/MV3_4stage_base_ablation/export_model.zip)|
|+VIM+StrideFormer+AAM|StrideFormer-Base|512x512|41.57%|265.5|5.62|[model](https://bj.bcebos.com/paddleseg/dygraph/ade20k/pp_mobileseg_base/model.pdparams)\|[log](https://bj.bcebos.com/paddleseg/dygraph/ade20k/pp_mobileseg_base/train.log)\|[vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/index?id=8b48fc0ada781be47468bdeb6941eb99)\|[exported model](https://bj.bcebos.com/paddleseg/dygraph/ade20k/pp_mobileseg_base/export_model.zip)|

\* Note that the latency is test with the final argmax operator using [PaddleLite](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/README_en.md) on xiaomi9 (Snapdragon 855 CPU) with single thread and 512x512 as input shape. Therefore the output of model is the segment result with single channel rather then probability logits. Inspired by the ineffectiveness of the final argmax operator that greatly increase the overall latency, we designed VIM to significantly decrease the latency.

\** The accuracy is reported based on self-trained reproduced result.

## <img src="https://user-images.githubusercontent.com/34859558/188439970-18e51958-61bf-4b43-a73c-a9de3eb4fc79.png" width="25"/> Reproduction

### Preparation
* Install PaddlePaddle and relative environments based on the [installation guide](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/install/pip/linux-pip_en.html).
* Install PaddleSeg based on the [reference](../../docs/install.md).
* Download the ADE20k dataset and link to PaddleSeg/data, or you can directly run, it will be automatically downloaded.

```
PaddleSeg/data
├── ADEChallengeData2016
│   ├── ade20k_150_embedding_42.npy
│   ├── annotations
│   ├── annotations_detectron2
│   ├── images
│   ├── objectInfo150.txt
│   └── sceneCategories.txt
```

### Training
You can start training by assign the ```tools/train.py``` with config files, the config files are under ```PaddleSeg/configs/pp_mobileseg```. Details about training are under [training guide](../../docs/train/train.md). You can find the trained models under ```Paddleseg/save/dir/best_model/model.pdparams```

```bash
export CUDA_VISIBLE_DEVICES=0,1

python3  -m paddle.distributed.launch tools/train.py \
    --config configs/pp_mobileseg/pp_mobileseg_base_ade20k_512x512_160k.yml \
    --save_dir output/pp_mobileseg_base \
    --save_interval 1000 \
    --num_workers 4 \
    --log_iters 100 \
    --use_ema \
    --do_eval \
    --use_vdl
```

### Validation
With the trained model on hand, you can verify the model's accuracy through evaluation. Details about evaluation are under [evaluation guide](../../docs/evaluation/evaluate.md).

```bash
python  -m paddle.distributed.launch tools/val.py --config configs/pp_mobileseg/pp_mobileseg_base_ade20k_512x512_160k.yml \
       --model_path output/pp_mobileseg_base/best_model/model.pdparams
```


### Deployment

We deploy the model on mobile devices for inference. To do that, we need to export the model and use [PaddleLite](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/README_en.md) to inference on mobile devices. You can also refer to [lite deply guide](../../docs/deployment/lite/lite.md) for details of PaddleLite deployment.

#### 0. Preparation
* An android mobile phone with usb debugger mode on and are already linked to your PC.
* Install the [adb tool](https://developer.android.com/studio/command-line/adb?hl=zh-cn).

Run the following command to make sure you are ready:

```bash
adb devices
# The following information will show if you are good to go:
List of devices attached
017QXM19C1000664    device
```

#### 1. Model exportation

The model need to be transfer from dynamic graph to static gradph for PaddleLite inference. In this step, we can use ```VIM``` to speed the model up. You only need to change ```model::upsample``` to ```vim``` in the config file and the exported model can be found on the `PaddleSeg/save/dir`

```bash
python tools/export.py --config configs/pp_mobileseg/pp_mobileseg_base_ade20k_512x512_160k.yml \
      --save_dir output/pp_mobileseg_base  \
      --input_shape 1 3 512 512 \ # The model is set to infer one image with this input shape, feel free to suit this to your dataset.
      --output_op none   # If do not use VIM, you need to set this to argmax to get the final prediction rather than logits.
```

#### 2. Model inference

* After the model is exported, you can download all the exported files and [tool zipfile](https://bj.bcebos.com/paddleseg/tools/test_tool.zip) as shown in the following file tree.

```markdown
Speed_test_dir
├── models_dir
│   ├── pp_mobileseg_base  # Files under this directory is generated through exportation
│   │   ├── model.pdmodel
│   │   ├── mdoel.pdiparams
│   │   ├── model.pdiparams.info
│   │   └── deploy.yaml
│   ├── pp_mobileseg_tiny
│   │   ├── model.pdmodel
│   │   ├── mdoel.pdiparams
│   │   ├── model.pdiparams.info
│   │   └── deploy.yaml
├── benchmark_bin   # The complied testscript of PaddleLite, which is in the tool zipfile.
├── image1.txt      # The txt file that stores the value of resized and normalized image
└── gen_val_txt.py  # You can use this script to generate the image1.txt for your test image
```

* And you can test the speed of the model using the following script. The tested result will be shown in the test_result.txt.
```bash
sh benchmark.sh benchmark_bin models_dir test_result.txt image1.txt
```

The test result on our PP-MobileSeg-Base is as following:
```markdown
-----------------Model=MV3_4stage_AAMSx8_valid_0321 Threads=1-------------------------
Delete previous optimized model: /data/local/tmp/seg_benchmark/models_0321/MV3_4stage_AAMSx8_valid_0321/opt.nb

---------- Opt Info ----------
Load paddle model from /data/local/tmp/seg_benchmark/models_0321/MV3_4stage_AAMSx8_valid_0321/model.pdmodel and /data/local/tmp/seg_benchmark/models_0321/MV3_4stage_AAMSx8_valid_0321/model.pdiparams
Save optimized model to /data/local/tmp/seg_benchmark/models_0321/MV3_4stage_AAMSx8_valid_0321/opt.nb

---------- Device Info ----------
Brand: Xiaomi
Device: cepheus
Model: MI 9
Android Version: 9
Android API Level: 28

---------- Model Info ----------
optimized_model_file: /data/local/tmp/seg_benchmark/models_0321/MV3_4stage_AAMSx8_valid_0321/opt.nb
input_data_path: /data/local/tmp/seg_benchmark/image1_norm.txt
input_shape: 1,3,512,512
output tensor num: 1
--- output tensor 0 ---
output shape(NCHW): 1 512 512
output tensor 0 elem num: 262144
output tensor 0 mean value: 1.18468e-44
output tensor 0 standard deviation: 2.52949e-44

---------- Runtime Info ----------
benchmark_bin version: e79b4b6
threads: 1
power_mode: 0
warmup: 20
repeats: 50
result_path:

---------- Backend Info ----------
backend: arm
cpu precision: fp32

---------- Perf Info ----------
Time(unit: ms):
init  = 33.071  
first = 314.619  
min   = 265.450  
max   = 271.217  
avg   = 267.246  
```

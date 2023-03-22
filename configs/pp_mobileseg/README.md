# PP-MobileSeg: Exploring Transformer Blocks for Efficient Mobile Segmentation.

## Reference
>

## Contents
1. Overview
2. Performance
3. Reproduction

## <img src="https://user-images.githubusercontent.com/34859558/190043857-bfbdaf8b-d2dc-4fff-81c7-e0aac50851f9.png" width="25"/> Overview
With the success of transformers in computer vision, several attempts have been made to adapt transformers to mobile devices. However, there still is room for optimization. Therefore, we propose PP-MobileSeg, a SOTA semantic segmentation model for mobile devices.

It is composed of three newly proposed parts, the mobilesegnet backbone, the Aggregated Attention Module(AAM) and the Valid Intepolate Moduel(VIM):
* With the four-stage mobilesegent as a efficient and effective backbone with little parameter overhead, we mange to extract rich semantic and detail features.
* To futher empower the feature, we use AAM to filter detail feature with ensemble voting and add the sematic feature in the end to presever it to the most content.
* At last, we use VIM to upsample the downsampled feature to the original resolution and greatly decrease latency. It only interpolates classes present in the final prediction which only takes around 10% in ADE20K dataset, which is also common for datasets with large classes. Therefore it greatly decrease the latency of the final upsample process which takes the greatest part of the model's overall latency.

Extensive experiments show that PP-MobileSeg achieves a superior params-accuracy-latency tradeoff compared to other SOTA methods.


## <img src="https://user-images.githubusercontent.com/34859558/190044217-8f6befc2-7f20-473d-b356-148e06265205.png" width="25"/> Performance
### ADE20K
| Model | Backbone | Training Iters | Train Resolution | Test Resolution | mIoU(%) | latency(ms)* | params(M) | Links |
|-|-|-|-|-|-|-|-|-|
|PP-MobileSeg-B|MobileSegNet-B|160000|512x512|512x512|41.24%|265.5|5.62|[config](./pp_mobileseg_base_ade20k_512x512_160k.yml)\|[model]()\|[log]()\|[vdl]()|
|PP-MobileSeg-T|MobileSegNet-T|160000|512x512|512x512|36.70%|215.3|1.61|[config](./pp_mobileseg_base_ade20k_512x512_160k.yml)\|[model]()\|[log]()\|[vdl]()|



### Ablation
| Model | Backbone | Training Iters | Train Resolution | Test Resolution | mIoU(%) | latency(ms)* | params(M) | Links |
|-|-|-|-|-|-|-|-|-|
|+VIM|Seaformer-B|160000|512x512|512x512|40.00%|234.6|8.17|[config](./pp_mobileseg_base_ade20k_512x512_160k.yml)\|[model]()\|[log]()\|[vdl]()|
|+VIM+MobileSegnet|MobileSegNet-B|160000|512x512|512x512|40.18%|235.1|5.54|[config](./pp_mobileseg_base_ade20k_512x512_160k.yml)\|[model]()\|[log]()\|[vdl]()|
|+VIM+MobileSegnet+AAM|MobileSegNet-B|160000|512x512|512x512|40.71%|265.5|5.62|[config](./pp_mobileseg_base_ade20k_512x512_160k.yml)\|[model]()\|[log]()\|[vdl]()|


### Compare with SOTA
| Model | Backbone | mIoU(%) | latency(ms)* | params(M) |
|-|-|-|-|-|
|LR-ASPP|MobileNetV3_large_x1_0|33.10|730.9|3.2|
|MobileSeg-Base|MobileNetV3_large_x1_0|33.26|391.5|2.85|
|TopFormer-Base|TopTransformer|38.28|480.6|5.13|
|SeaFormer-Base**|SeaFormer|40.0|465.4|8.64|
|PP-MobileSeg-Base|MobileSegNet-B|41.24|265.5|5.62|
|PP-MobileSeg-Tiny|MobileSegNet-B|35.71|215.3|1.61|


\* The latency is test with the final argmax operator on Snapdragon 855.

\** The accuracy is reported based on self-trained reproduced result.

## <img src="https://user-images.githubusercontent.com/34859558/188439970-18e51958-61bf-4b43-a73c-a9de3eb4fc79.png" width="25"/> Reproduction

### Preparation
* Install PaddlePaddle and relative environments based on the [installation guide](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/install/pip/linux-pip_en.html)
* Install PaddleSeg based on the [reference](../../docs/install.md)
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
You can start training by assign the train.py with config files, the config files are under PaddleSeg/configs/pp_mobileseg. Details about training are under [training guide](../../docs/train/train.md). You can find the trained models under Paddleseg/save/dir/best_model/model.pdparams

```bash
export CUDA_VISIBLE_DEVICES=0,1;

python3  -m paddle.distributed.launch tools/train.py \
    --config configs/pp_mobileseg/pp_mobileseg_base_ade20k_512x512_160k.yml --use_ema \
    --do_eval  --use_vdl --save_interval 1000 --save_dir output/pp_mobileseg_base --num_workers 4  --log_iters 100
```

### Validation
With the trained model on hand, you can verify the model's accuracy through evaluation. Details about evaluation are under [evaluation guide](../../docs/evaluation/evaluate.md).

```bash
python tools/val.py --config configs/pp_mobileseg/pp_mobileseg_base_ade20k_512x512_160k.yml \
       --model_path output/pp_mobileseg_base/best_model/model.pdparams \
       # the configs below this line is augmentation during evaluation which is not used in our reported result.
       --aug_eval \  
       --scales 0.75 1.0 1.25 \
       --flip_horizontal
```


### Deployment

We deploy the model on mobile devices for inference. To do that, we need to export the model and use [PaddleLite](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/README_en.md) to inference on mobile devices. You can also refer to [lite deply guide](../../docs/deployment/lite/lite.md) for details of PaddleLite deployment.

#### 0. Preparation
* An android mobile phone with usb debugger mode on and already link to your PC.


#### 1. Model exportation

The model will be transfer from dynamic graph to static gradph for inference speedup. You can find the exported model on the `PaddleSeg/save/dir`

```bash
python tools/export.py --config configs/pp_mobileseg/pp_mobileseg_base_ade20k_512x512_160k.yml \
      --save_dir output/pp_mobileseg_base  \
      --input_shape 1 3 512 512 \ # The model is set to infer one image with this input shape, feel free to suit this to your dataset.
      --output_op none   # If do not use VIM, you need to set this to argmax to get the final prediction rather than logits.
```

#### 2. Model inference

* After the model is exported, you can download all the exported files and [tool zip](https://bj.bcebos.com/paddleseg/tools/test_tool.zip) as shown in the following file tree.

```markdown
Speed_test_dir
├── models_dir
│   ├── pp_mobileseg_base  # Files under this directory will be
│   │   ├── model.pdmodel
│   │   ├── mdoel.pdiparams
│   │   ├── model.pdiparams.info
│   │   └── deploy.yaml
│   ├── pp_mobileseg_tiny
│   │   ├── model.pdmodel
│   │   ├── mdoel.pdiparams
│   │   ├── model.pdiparams.info
│   │   └── deploy.yaml
├── benchmark_bin   # The complied testscript of PaddleLite which is in the tool zip.
├── image1.txt      # The txt file that stores the value of resized and normalized image
└── gen_val_txt.py  # You can use this script to generate the image1.txt
```

* And you can test the speed of the model using the following script. The tested result will be shown in the test_result.txt.
```bash
sh benchmark.sh benchmark_bin models_dir test_result.txt image1.txt
```

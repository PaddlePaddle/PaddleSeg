# Cityscapes SOTA
The implementation of Hierarchical Multi-Scale Attention based on PaddlePaddle. [[Paper]](https://arxiv.org/abs/2005.10821)<br>

Based on the above work, we made some optimizations:
- Use dice loss and bootstrapped cross entropy loss instead of cross entropy
- Learn all fine data and equal amount of coarse data in each epoch
- The evaluation is carried out by using the equal difference scale series instead of the equal ratio scale series

We achieve mIoU of **87%** on Cityscapes validation set.

The actual effect is as follows (for high-definition pictures, please click [here](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0/docs/images/cityscapes.gif)).

<div align="center">
<img src=https://user-images.githubusercontent.com/30695251/144982303-51d40188-c00d-46b7-9012-41955c4e2156.gif  width = "500" />  
</div>

## Installation

#### step 1. Install PaddlePaddle

System Requirements:
* PaddlePaddle >= 2.0.0rc1
* Python >= 3.6+

Highly recommend you install the GPU version of PaddlePaddle, due to large overhead of segmentation models, otherwise it could be out of memory while running the models. For more detailed installation tutorials, please refer to the official website of [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/)。


#### step 2. Install PaddleSeg

You should use *API Calling* method to install PaddleSeg for flexible development.

```shell
pip install paddleseg
```

## Data Preparation
Download following files and put into `data/cityscapes` directory. Then unzip these files.
```shell
mkdir -p data/cityscapes
```

Firstly please download 3 files from [Cityscapes dataset](https://www.cityscapes-dataset.com/downloads/)
- leftImg8bit_trainvaltest.zip (11GB)
- gtFine_trainvaltest.zip (241MB)
- leftImg8bit_trainextra.zip (44GB)

Run the following commands to do the label conversion:
```shell
pip install cityscapesscripts
python ../../tools/convert_cityscapes.py --cityscapes_path data/cityscapes --num_workers 8
```
Where 'cityscapes_path' should be adjusted according to the actual dataset path. 'num_workers' determines the number of processes started and the size can be adjusted according to the actual situation.

Then download and uncompress Autolabelled-Data from [google drive](https://drive.google.com/file/d/1DtPo-WP-hjaOwsbj6ZxTtOo_7R_4TKRG/view?usp=sharing)
- refinement_final_v0.zip # This file is needed for autolabelled training for recreating SOTA

Delete useless `tmp` directory in `refinement_final` directory.
```
rm -r tmp/
```

Convert autolabelled data according to PaddleSeg data format:
```shell
python tools/convert_cityscapes_autolabeling.py --dataset_root data/cityscapes/
```

Finally, you need to organize data following the below structure.

    cityscapes
    |
    |--leftImg8bit
    |  |--train
    |  |--val
    |  |--test
    |
    |--gtFine
    |  |--train
    |  |--val
    |  |--test
    |
    |--leftImg8bit_trainextra
    |  |--leftImg8bit
    |     |--train_extra
    |        |--augsburg
    |        |--bayreuth
    |        |--...
    |
    |--convert_autolabelled
    |  |--augsburg
    |  |--bayreuth
    |  |--...



## Evaluation

### Download Trained Model
```shell
mkdir -p saved_model && cd saved_model
wget https://bj.bcebos.com/paddleseg/dygraph/cityscapes/mscale_ocr_hrnetw48_cityscapes_autolabel_mapillary/model.pdparams
cd ..
```
### Evaluation on Cityscapes

| Model | Backbone | mIoU | mIoU (flip) | mIoU (5 scales + flip) |
|:-:|:-:|:-:|:-:|:-:|
|MscaleOCRNet|HRNet_w48|86.89%|86.99%|87.00%|

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m paddle.distributed.launch val.py \
--config configs/mscale_ocr_cityscapes_autolabel_mapillary.yml --num_workers 3 --model_path saved_model/model.pdparams
```
The reported mIoU should be 86.89. This evaluates with scales of 0.5, 1.0 and 2.0. This requires 14.2GB of GPU memory.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m paddle.distributed.launch val.py \
--config configs/mscale_ocr_cityscapes_autolabel_mapillary.yml --num_workers 3 --model_path saved_model/model.pdparams \
--aug_eval --flip_horizontal
```
The reported mIoU should be 86.99. This evaluates with scales of 0.5, 1.0, 2.0 and flip horizontal. This requires 14.2GB of GPU memory.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m paddle.distributed.launch val.py \
--config configs/mscale_ocr_cityscapes_autolabel_mapillary_ms_val.yml --num_workers 3 --model_path saved_model/model.pdparams \
--aug_eval --flip_horizontal
```
The reported mIoU should be 87.00. This evaluates with scales of 0.5, 1.0, 1.5, 2.0, 2.5 and flip horizontal. This requires 21.2GB of GPU memory.

## Training
### Download Pretrained Weights

```shell
mkdir -p pretrain && cd pretrain
wget https://bj.bcebos.com/paddleseg/dygraph/cityscapes/ocrnet_hrnetw48_mapillary/pretrained.pdparams
cd ..
```

Pretrained weights were obtained by pretraining on the Mapillary dataset from OCRNet (backbone is HRNet w48).

### Training on Cityscapes
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m paddle.distributed.launch train.py \
--config configs/mscale_ocr_cityscapes_autolabel_mapillary.yml --use_vdl \
--save_dir saved_model/mscale_ocr_cityscapes_autolabel_mapillary --save_interval 2000 --num_workers 5 --do_eval
```
Note that this requires 32GB of GPU memory. You can remove argument `--do_eval` to turn off evaluation during training, thus it only requires 25GB of GPU memory.
If you run out of memory, try to lower the crop size.

## Deploy

- Inference with TensorRT in C++
- Inference with ONNX Runtime in Python
- Inference with TensorFlow Lite in Python

Please refer to
- https://github.com/PINTO0309/PINTO_model_zoo/tree/main/201_CityscapesSOTA/demo
- https://github.com/iwatake2222/play_with_tensorrt/tree/master/pj_tensorrt_seg_paddleseg_cityscapessota
- https://github.com/axinc-ai/ailia-models/tree/master/image_segmentation/paddleseg

Thanks for their contributions!

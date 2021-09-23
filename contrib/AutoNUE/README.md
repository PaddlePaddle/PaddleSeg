# AutoNUE@CVPR 2021 Challenge
Implementation of the 1st solution for AutoNUE@CVPR 2021 Challenge Semenatic Segmentation Track based on PaddlePaddle.

## Installation

#### step 1. Install PaddlePaddle

System Requirements:
* PaddlePaddle >= 2.0.0
* Python >= 3.6+

Highly recommend you install the GPU version of PaddlePaddle, due to large overhead of segmentation models, otherwise it could be out of memory while running the models. For more detailed installation tutorials, please refer to the official website of [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/)。


#### step 2. Install PaddleSeg

You should use *API Calling* method to install PaddleSeg for flexible development.

```shell
pip install paddleseg -U
```

## Data Preparation

Firstly, you need to to download and convert the [India Driving Dataset](https://idd.insaan.iiit.ac.in/evaluation/autonue21/#bm5) following the instructions of Segmentation Track. IDD_Dectection dataset also need for pseudo-labeling.

And then, you need to organize data following the below structure.

    IDD_Segmentation
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

We make three contributions and managed to rank 1st.
- Progressively Segmentation
- Leverage IDD_Detection Dataset to generate extre training samples by pseudo-labeling.
- Decoder-enhanced Swin Transformer

## Training

### Baseline
1. Download pretrained weights on Mapillary.

```shell
mkdir -p pretrain && cd pretrain
wget https://bj.bcebos.com/paddleseg/dygraph/cityscapes/ocrnet_hrnetw48_mapillary/pretrained.pdparams
cd ..
```
2. Modify `scripts/train.py` line 27 with `from core.val import evaluate`
3. Run the training script.
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m paddle.distributed.launch train.py \
--config configs/sscale_auto_nue_map+city@1920.yml --use_vdl \
--save_dir saved_model/sscale_auto_nue_map+city@1920 --save_interval 2000 --num_workers 2 --do_eval
```

### Regional progressive segmentation
1. Replace `scripts/train.py` line 27 'from core.val import evaluate' with `from core.val_crop import evaluate`
2. Run
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m paddle.distributed.launch train.py \
--config configs/auto_nue_map+city_crop.yml --use_vdl \
--save_dir saved_model/auto_nue_map+city_crop --save_interval 2000 --num_workers 2 --do_eval
```

### Pseudo-labeling
First you need to organize the IDD_Detection dataset as follow:


    IDD_Detection
    |
    |--JPEGImages
    |--Annotations


where `JPEGImages` and `Annotation` are images and xml files collected from `IDD_Detection/FrontFar` and `IDD_Detection/FrontNear` two folders.

And Then:
1. Replace `AutoNUE21/predict.py` line 22 `from paddleseg.core import predict` with `from core.predict_generate_autolabel.py import predictAutolabel`
2. Modity `AutoNUE21/predict.py` line 156 `predict(` with `predictAutolabel(`
3. Run
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m paddle.distributed.launch  predict.py --config configs/sscale_auto_nue_map+city@1920.yml  --model_path saved_model/sscale_auto_nue_map+city@1920/best_model/model.pdparams --image_path data/IDD_Detection/JPEGImages --save_dir detection_out --aug_pred --scales 1.0 1.5 2.0 --flip_horizontal
```
4. Auto-box `traffic lights` and `traffic sign` two classes from bounding box annotation by running `tools/IDD_labeling.py`
5. Put the generated `pred_refine` folder under `data/IDD_Detection`
5. Modify `scripts/train.py` line 27 with `from core.val import evaluate`
6. Train these pseudo labels with fine-annotated sample:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m paddle.distributed.launch train.py \
--config configs/auto_nue_auto_label.yml --use_vdl \
--save_dir saved_model/auto_nue_auto_label --save_interval 2000 --num_workers 2 --do_eval
```

### Decoder-enhanced Swin Transformer

1. Download pretrained weights on Mapillary.

```shell
cd pretrain
wget https://bj.bcebos.com/paddleseg/dygraph/cityscapes/swin_mla_p4w7_mapillary/pretrained_swin.pdparams
cd ..
```

2. Run the training script.
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m paddle.distributed.launch train.py \
--config configs/swin_transformer_mla_base_patch4_window7_160k_autonue.yml --use_vdl \
--save_dir saved_model/swin_transformer_mla_autonue --save_interval 2000 --num_workers 2 --do_eval
```
3. Run the testing script.
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m paddle.distributed.launch  predict.py --config configs/swin_transformer_mla_base_patch4_window7_160k_autonue.yml  --model_path saved_model/swin_transformer_mla_autonue/best_model/model.pdparams --image_path data/IDD_Segmentation/leftImg8bit/test/ --save_dir test_out_swin --aug_pred --scales 1.0 1.5 2.0 --flip_horizontal
```

## Ensemble Testing
We provide a predict script for ensembling `baseline`, `pseudo-labeling` and `rps`.
Just running:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m paddle.distributed.launch  predict_ensemble_three.py --config configs/sscale_auto_nue_map+city@1920.yml  --config_1 configs/auto_nue_auto_label.yml --config_crop configs/auto_nue_map+city_crop.yml --model_path saved_model/sscale_auto_nue_map+city@1920/best_model/model.pdparams  --model_path_1 saved_model/auto_nue_auto_label/best_model/model.pdparams  --model_path_crop saved_model/auto_nue_map+city_crop/best_model/model.pdparams --image_path data/IDD_Segmentation/leftImg8bit/test/ --save_dir test_out --aug_pred --scales 1.0 1.5 2.0 --flip_horizontal
```

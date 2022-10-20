[简体中文](./quick_start_cn.md) | English

# Quick Start

This document uses a simple example to introduce the usage of PaddleSeg.

At first, please refer to the [installation](./install.md) to prepare the develop environment.

## 1 Prepare Data

Our demo use the optic disc segmentation dataset, as shown in the next figure.

<div align="center">
<img src="./images/fig1.png"  width = "400" />  
</div>

Run the following command to download the dataset, and save it in `PaddleSeg/data`.

```
cd PaddleSeg
mkdir data
cd data
wget https://paddleseg.bj.bcebos.com/dataset/optic_disc_seg.zip
unzip optic_disc_seg.zip
cd ..
```


## 2 Prepare Config File

Usually, we use the config file in model training, validation and deployment.

The config file contains all necessary information, such as segmentation model, loss, training dataset, validation dataset, etc.

The config file of our demo is `PaddleSeg/configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml`


## 3 Model Training

Run the following command in the root directory of PaddleSeg to start training model with single GPU.


```
export CUDA_VISIBLE_DEVICES=0 # Use single GPU in Linux
# set CUDA_VISIBLE_DEVICES=0 # Use single GPU in Windows

python tools/train.py \
       --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
       --save_interval 500 \
       --do_eval \
       --use_vdl \
       --save_dir output
```

After the training, the mIoU is 90.65% for the validation set, and the trained weight is saved in `PaddleSeg/output/`.


## 4 Model Validation

In the root directory of PaddleSeg, run the following command to evaluate the trained model on the validation set.


```
python tools/val.py \
       --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
       --model_path output/best_model/model.pdparams
```


## 5 Model Prediction

In the root directory of PaddleSeg, run the following command to load trained model, predict the segmentation result and save the result image.

```
python tools/predict.py \
       --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
       --model_path output/best_model/model.pdparams \
       --image_path data/optic_disc_seg/JPEGImages/H0002.jpg \
       --save_dir output/result
```

The result image is save in `PaddleSeg/output/result`, as shown in next figure.


<div align="center">
<img src="./images/fig5.png"  width = "600" />  
</div>

## 6 Others

Based on the trained model, we can export the inference model and deploy it on various devices. Please refer to the PaddleSeg documents for detailed description.

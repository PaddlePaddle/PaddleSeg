# Unsupervised Domain Adaptation on Semantic Segmentation

Domain adpataion is a kind of algorithms that dedicate to minimize the performance drop when we apply the model trained from one domain on anther. Normally, this happens when we want to apply models which is trained with dataset A on a similar dataset B. Specifically, Unsupervised domain adaptation(UDA) dedicate to achieve satisfactory performance on dataset B that we do not have label on. Following image from [pixmatch](https://arxiv.org/abs/2105.08128) shows the performance of UDA algorithms on semantic segmentaion. When we compare the segmentaiton results between "source only"(without DA) and "ours", we can observe a great performance lift.

![image-20211116171626323](https://github.com/PaddlePaddle/PaddleSeg/blob/c5f21a1a61ce7be926284a5c3920318ff2f95356/contrib/DomainAdaptation/docs/domain_adaptation.png)

This project includes the reproduction of pixmatch [[Paper](https://arxiv.org/abs/2105.08128)|[Code](https://github.com/lukemelas/pixmatch)] based on PaddlePaddle and PaddleSeg, which reaches mIOU = 47.8% on Cityscapes Dataset.

In addition to reproduce pixmatch, we also tried something new. Major adjustment including:

1. Add edge constrain branch to improve edge accuracy (negative results, still needs to adjust)
2. Use edge as prior information to fix segmentation result (negative results, still needs to adjust)
3. Align features' structure across domain (positve result, reached mIOU=48.0%)

## Model performance

|        Model        | Backbone  | Resolution | Training Iters | mIoU  |                            Links                             |
| :-----------------: | :-------: | :--------: | :------------: | :---: | :----------------------------------------------------------: |
|      PixMatch       | resnet101 |  1280x640  |     60000      | 47.8% | [model](https://bj.bcebos.com/paddleseg/domain_adaptation/gta5_to_cityscapes/pixmatch/model.pdparams) \|[log](https://bj.bcebos.com/paddleseg/domain_adaptation/gta5_to_cityscapes/pixmatch/train.log) \|[vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/index?id=432b12ae9a79eedd0db35277835ee419) |
| Pixmatch-featpullin | resnet101 |  1280x640  |     100000     | 48.0% | [model](https://bj.bcebos.com/paddleseg/domain_adaptation/gta5_to_cityscapes/pixmatch_featpullin/model.pdparams) \|[log](https://bj.bcebos.com/paddleseg/domain_adaptation/gta5_to_cityscapes/pixmatch_featpullin/train.log) \|[vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=77c2de204c9fb00844afb88dd42ba365) |

If you would like to try out our project, there are serveral things you need to figure out.

## Few things to do before try out

### 1. Install environment

```
git clone https://github.com/PaddlePaddle/PaddleSeg.git
cd contrib/DomainAdaptation/
pip install -r requirments.txt
python -m pip install paddlepaddle-gpu==2.2.0 -i https://mirror.baidu.com/pypi/simple
```

### 2. Download Datasets & pretrained model

1. Download [GTA5 dataset](https://download.visinf.tu-darmstadt.de/data/from_games/),  [cityscapes dataset](https://www.cityscapes-dataset.com/) and relative [data_list](https://github.com/lukemelas/pixmatch/tree/master/datasets).

   1. Orgranize the dataset and data list as following:

      ```
      data
      ├── cityscapes
      │   ├── gtFine
      │   │   ├── train
      │   │   │   ├── aachen
      │   │   │   └── ...
      │   │   └── val
      │   └── leftImg8bit
      │       ├── train
      │       └── val
      ├── GTA5
      │   ├── images
      │   ├── labels
      │   └── list
      ├── city_list
      └── gta5_list
      ```

2. Download [pretrained model](https://bj.bcebos.com/paddleseg/domain_adaptation/pretrained//gta5_pretrained.pdparams) on GTA5 and save it to models/.

### 3. Train and test

1. Train on one GPU

   1. Try the project as the reproduction of pixmatch:  `sh run-DA_src.sh`
   2. Try the project for other experiment:  change to another config file in the training script

2. Validate on one GPU:

   1. Download the [trained model](https://bj.bcebos.com/paddleseg/domain_adaptation/gta5_to_cityscapes/pixmatch/model.pdparams) on cityscapes, and save it to models/model.pdparams:

   2. Validate with following script. Since we forget to save the ema model, the validation result is 46% :

      ```
      python3 -m paddle.distributed.launch val.py --config configs/deeplabv2_resnet101_os8_gta5cityscapes_1280x640_160k_newds_gta5src.yml --model_path models/model.pdparams --num_workers 4
      ```

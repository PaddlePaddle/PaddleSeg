# ps-mt-paddleseg

**[CVPR'22]**[[Perturbed and Strict Mean Teachers for Semi-supervised Semantic Segmentation]](https://arxiv.org/abs/2111.12903)

by Yuyuan Liu, [Yu Tian](https://yutianyt.com/), Yuanhong Chen, [Fengbei Liu](https://fbladl.github.io/), [Vasileios Belagiannis](https://campar.in.tum.de/Main/VasileiosBelagiannis) and [Gustavo Carneiro](https://cs.adelaide.edu.au/~carneiro/)

Computer Vision and Pattern Recognition Conference (CVPR), 2022

![image](https://user-images.githubusercontent.com/102338056/167279043-362e1405-db45-4355-b92b-0993312fe461.png)

In this project, we reproduce PS-MT [[Paper](https://arxiv.org/abs/2111.12903)|[Code](https://github.com/yyliu01/PS-MT)] with PaddlePaddle and reaches mIOU = 78.05% on Pascal VOC12 Dataset.

## Model performance 

| Model | Backbone  | Resolution | Training Iters | mIoU  |                            Links                             |
| :---: | :-------: | :--------: | :------------: | :---: | :----------------------------------------------------------: |
| PS-MT | resnet101 |  512*512   |     20000      | 78.05 | [model](链接: https://pan.baidu.com/s/1wWYxRdMSDvET2dUo1cxXgA?pwd=qyvp 提取码: qyvp 复制这段内容后打开百度网盘手机App，操作更方便哦)\|[log](https://wandb.ai/ps-mt-2022/PS-MT(VOC12)?workspace=user-2275029710) |

## Prepare Work

### 1. Install environment

```
git clone https://github.com/PaddlePaddle/PaddleSeg.git
cd contrib/ps-mt-paddleseg/
pip install -r requirments.txt
python -m pip install paddlepaddle-gpu==2.3.0 -i https://mirror.baidu.com/pypi/simple
```

### 2. Download Datasets & pretrained model

1. Download [origion dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar),[SegmentationClassAug](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0)
2. Download [pretrained model](https://onedrive.live.com/redir?resid=B71317D47B7AC1CB!895&authkey=!AGEiz96zF_Rougc&e=c5cZvF) on Pascal VOC12 and save it to paddleseg/models/.
2. you should put the files in voc_split into /VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation

### 3. Train and test

1. Train on one GPU

   ```
   python train.py
   ```

2. predict on one GPU 

   1. Download the  [trained model](链接: https://pan.baidu.com/s/1wWYxRdMSDvET2dUo1cxXgA?pwd=qyvp 提取码: qyvp 复制这段内容后打开百度网盘手机App，操作更方便哦) on Pascal VOC12, and save it to model.pdparams

   ```
   python predict.py
   ```

   

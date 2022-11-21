# Dataset Preparation

When training for custom dataset, you need to prepare dataset in the appropriate format.
We provide two forms of dataset, one for offline composition and one for online composition.

## Offline composition
If the images have been composited offline or do not need to be composited, the dataset should be organized as follows.
```
dataset_root/
|--train/
|  |--fg/
|  |--alpha/
|
|--val/
|  |--fg/
|  |--alpha/
|
|--train.txt
|
|--val.txt
```
where, fg folder stores the original images. The image name in the fg folder must correspond to that in the alpha folder one by one,
and the resolution must be the same for the correspond image in the two folders.

train.txt and val.txt contents are as follows.
```
train/fg/14299313536_ea3e61076c_o.jpg
train/fg/14429083354_23c8fddff5_o.jpg
train/fg/14559969490_d33552a324_o.jpg
...
```

## Online composition
Data reading support online composition, that is, the image input network composited online by the foreground, alpha, and background images,
which like the Composition-1k dataset used in [Deep Image Matting](https://arxiv.org/pdf/1703.03872.pdf) .
The dataset should be organized as follows:
```
Composition-1k/
|--bg/
|
|--train/
|  |--fg/
|  |--alpha/
|
|--val/
|  |--fg/
|  |--alpha/
|  |--trimap/ (如果存在)
|
|--train.txt
|
|--val.txt
```

where, the fg folder stores the foreground images and the bg folder stores the background images.

The contents of train.txt is as follows:
```
train/fg/fg1.jpg bg/bg1.jpg
train/fg/fg2.jpg bg/bg2.jpg
train/fg/fg3.jpg bg/bg3.jpg
...
```
where, the first column is the foreground images and the second column is the background images.

The contents of val.txt are as follows. If trimap does not exist in dataset, the third column is not needed and the code will generate trimap automatically.
```
val/fg/fg1.jpg bg/bg1.jpg val/trimap/trimap1.jpg
val/fg/fg2.jpg bg/bg2.jpg val/trimap/trimap2.jpg
val/fg/fg3.jpg bg/bg3.jpg val/trimap/trimap3.jpg
...
```

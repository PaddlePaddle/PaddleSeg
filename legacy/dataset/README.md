# 数据下载
## PASCAL VOC 2012数据集
下载 PASCAL VOC 2012数据集并将分割部分的假彩色标注图(`SegmentationClass`文件夹)转换成灰度图并存储在`SegmentationClassAug`文件夹, 并在文件夹`ImageSets/Segmentation`下重新生成列表文件`train.list、val.list和trainval.list。

```shell
# 下载数据集并进行解压转换
python download_and_convert_voc2012.py
```

如果已经下载好PASCAL VOC 2012数据集，将数据集移至dataset目录后使用下述命令直接进行转换即可。

```shell
# 数据集转换
python convert_voc2012.py
```

## Oxford-IIIT Pet数据集
我们使用了Oxford-IIIT中的猫和狗两个类别数据制作了一个小数据集mini_pet，更多关于数据集的介绍请参考[Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)。

```shell
# 下载数据集并进行解压
python dataset/download_pet.py
```

## Cityscapes数据集
运行下述命令下载并解压Cityscapes数据集。

```shell
# 下载数据集并进行解压
python dataset/download_cityscapes.py
```

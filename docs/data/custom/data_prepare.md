# 自定义数据集

如果您需要使用自定义数据集进行训练，请按照以下步骤准备数据.

1.推荐整理成如下结构

    custom_dataset
        |
        |--images
        |  |--image1.jpg
        |  |--image2.jpg
        |  |--...
        |
        |--labels
        |  |--label1.png
        |  |--label2.png
        |  |--...
        |
        |--train.txt
        |
        |--val.txt
        |
        |--test.txt

其中train.txt和val.txt的内容如下所示：

    images/image1.jpg labels/label1.png
    images/image2.jpg labels/label2.png
    ...

2.标注图像的标签从0,1依次取值，不可间隔。若有需要忽略的像素，则按255进行标注。

可按如下方式对自定义数据集进行配置：
```yaml
train_dataset:
  type: Dataset
  dataset_root: custom_dataset
  train_path: custom_dataset/train.txt
  num_classes: 2
  transforms:
    - type: ResizeStepScaling
      min_scale_factor: 0.5
      max_scale_factor: 2.0
      scale_step_size: 0.25
    - type: RandomPaddingCrop
      crop_size: [512, 512]
    - type: RandomHorizontalFlip
    - type: Normalize
  mode: train
```
请注意**数据集路径和训练文件**的存放位置，按照代码中的dataset_root和train_path示例方式存放。

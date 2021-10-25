# Tools

## eiseg2paddlex

在使用EISeg对网络爬取的图像标注完成后，通过`tool`中的`eiseg2paddlex`，可以将EISeg标注好的数据快速转换为PaddleX的训练格式。

### 使用

可以使用以下方法进行转换：

```
python eiseg2paddlex.py -d save_folder_path -o image_folder_path [-l label_folder_path] [-s split_rate]
```
其中:
- `save_folder_path`: 为需要保存PaddleX数据的路径，必填
- `image_folder_path`: 为图像的路径，必填
- `label_folder_path`: 为标签的路径，非必填，若不填则为自动保存的位置（`image_folder_path/label`）
- `split_rate`: 训练集和验证集划分的比例，非必填，若不填则为0.9

### 效果
![](https://s3.bmp.ovh/imgs/2021/10/714c439a9c7fa49b.png)

### PaddleX

#### 训练

``` python
! pip -q install paddlex
# ! mkdir -p MyDataset
# ! unzip -oq work/pdx_datas.zip -d MyDataset

import paddlex as pdx
from paddlex import transforms as T

train_transforms = T.Compose([
    T.Resize(target_size=512),
    T.RandomHorizontalFlip(),
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

eval_transforms = T.Compose([
    T.Resize(target_size=512),
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_dataset = pdx.datasets.SegDataset(
                        data_dir='./MyDataset',
                        file_list='./MyDataset/train_list.txt',
                        label_list='./MyDataset/labels.txt',
                        transforms=train_transforms)
eval_dataset = pdx.datasets.SegDataset(
                        data_dir='./MyDataset',
                        file_list='./MyDataset/val_list.txt',
                        label_list='MyDataset/labels.txt',
                        transforms=eval_transforms)

num_classes = len(train_dataset.labels)
model = pdx.seg.BiSeNetV2(num_classes=num_classes)

model.train(
    num_epochs=5,
    train_dataset=train_dataset,
    train_batch_size=4,
    eval_dataset=eval_dataset,
    learning_rate=0.05,
    save_dir='output/bisenet')
```

#### 预测

``` python
import matplotlib.pyplot as plt
%matplotlib inline

model = pdx.load_model('output/bisenet/epoch_5')
image_name = 'MyDataset/JPEGImages/u_3094633203_306616428&fm_253&fmt_auto&app_120&f_JPEG.jpg'
result = model.predict(image_name)
pred = pdx.seg.visualize(image_name, result, weight=0.3, save_dir=None)
plt.imshow(pred)
```
![](https://s3.bmp.ovh/imgs/2021/10/5191b28b4b1ab747.png)
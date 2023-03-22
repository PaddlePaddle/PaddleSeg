# 无监督异常检测工具

## 1. 环境依赖

* python
* paddlepaddle-gpu
* tqdm
* sklearn
* matplotlib


## 2. 数据集

此工具以MVTec AD数据集为例, [下载MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/);


## 3. 训练, 评估, 预测命令

Train:

```python tools/uad/train.py --config /ssd1/zhaoyantao/PP-Industry/configs/uad/padim/padim_resnet18_mvtec.yml --category bottle```

Eval:

```python tools/uad/val.py --config /ssd1/zhaoyantao/PP-Industry/configs/uad/padim/padim_resnet18_mvtec.yml --category bottle```

Predict:

```python tools/uad/predict.py --config /ssd1/zhaoyantao/PP-Industry/configs/uad/padim/padim_resnet18_mvtec.yml --category bottle```



## 4. 配置文件解读

无监督异常检测(uad)模型的配置文件主要包含以下参数:

```
# common arguments
device: gpu
batch_size: 1
seed: 3   # 指定numpy, paddle的随机种子

category: leather # 指定MVTecAD数据集的某个类别
data_path: /ssd1/zhaoyantao/PP-Industry/data/mvtec_anomaly_detection  # 指定MVTecAD数据集的根目录
save_path: /ssd1/zhaoyantao/PP-Industry/output    # 指定训练时模型参数保存的路径和评估/预测时结果图片的保存路径

# train arguments
do_val: True  # 指定训练时是否进行评估
backbone_depth: 18    # 指定resnet骨干网络的深度
pretrained_backbone: True # 指定骨干网络是否加载imagenet预训练权重

# val and predict arguments
save_picture: False   # 指定是否保存第一张评估图片/预测图片的结果
model_path: /ssd1/zhaoyantao/PP-Industry/output/leather/best.pdparams # 指定加载模型参数的路径

img_path: /ssd1/zhaoyantao/PP-Industry/data/mvtec_anomaly_detection/cable/test/bent_wire/000.png  # 指定预测的图片路径
threshold: 0.5    # 指定预测后二值化异常分数图的阈值
```

## 5. 集成模型

目前uad工具集成了[PaDiM](../../configs/uad/padim/README.md)模型;

# 快速开始
本文档将以Magnetic-tile-defect-datasets数据，并选用目标检测+RoI分割+后处理的串联解决方案为例，展示如何快速上手QualityInspector工具。

## 1 安装说明

请参考：[详细教程——安装说明](./install.md)
执行完毕后，当前目录位于`PaddleSeg/contrib/QualityInspector/`

## 2 准备数据集

2.1 下载：在当前目录下建立dataset文件夹，请前往[Magnetic-tile-defect-datasets](https://github.com/abin24/Magnetic-tile-defect-datasets.)下载数据集，放在`./dataset/`路径下。具体可见[详细教程——数据准备1.1](./tools_data/prepare_data.md#11-公开数据集下载)

2.2 预处理：为了满足PaddleSeg训练支持的分割数据格式，首先将数据进行预处理，具体可见[详细教程——数据准备1.2](./tools_data/prepare_data.md#12-转换为全图分割训练数据格式).

```shell
python3 tools/dataset/MT_dataset.py --dataset_path ./dataset/Magnetic-Tile-Defect --output_path ./dataset/MT_dataset/
```

2.3 数据格式转化：若希望尝试检测+RoI分割的解决方案，需要对2.2得到的分割数据进行数据格式转换，转化为目标检测PaddleDetection支持的训练数据格式，具体可见[详细教程——数据准备1.4](./tools_data/prepare_data.md#14-将全图分割数据转为coco格式的json文件)，

```shell
python3 tools/convert_tools/convert_mask_to_coco.py --image_path dataset/MT_dataset/images/train --anno_path dataset/MT_dataset/annos/train --class_num 5 --label_file dataset/MT_dataset/mt_catIDs.json --output_name dataset/MT_dataset/train.json --suffix .png
```

执行后json文件保存在`dataset/MT_dataset/train.json`，将上述命令`--image_path`、`--anno_path`和`--output_name`输入的路径中的`train`改为`val`在执行，得到`dataset/MT_dataset/val.json`

同时需要切割RoI区域用于分割训练，具体可见[详细教程——数据准备1.3](./tools_data/prepare_data.md#13-将全图分割数据转为roi分割文件).

```shell
python3 tools/convert_tools/convert_mask_to_roi.py --image_path dataset/MT_dataset/images/train --anno_path dataset/MT_dataset/annos/train --class_num 5 --output_path dataset/MT_dataset/RoI/train/ --suffix .png --to_binary
```
执行后数据保存在`dataset/MT_dataset/RoI/train/`，同样`train`改为`val`得到验证集RoI分割数据.

## 3 训练和验证
在安装和准备数据完成后，即可进行模型的训练。具体可参考[详细教程——训练&验证3](./det_seg/train_eval.md#训练).

3.1 模型训练：

将数据路径写到config中，分别训练目标检测模型和RoI分割模型：

检测：
```bash
python3 tools/det/train.py -c configs/det/hrnet/faster_rcnn_hrnetv2p_w18_3x_defect.yml -o weights=./output/faster_rcnn_hrnetv2p_w18_3x_defect/model_final.pdparams
```

RoI分割：

```bash
python3 tools/seg/train.py --config configs/seg/ocrnet/ocrnet_hrnetw18_RoI_defect_256x256_40k.yml --do_eval  --use_vdl --save_interval 100 --save_dir ./output/RoI/
```
其中配置文件`configs/seg/ocrnet/ocrnet_hrnetw18_RoI_defect_256x256_40k.yml`中`train_path: dataset/MT_dataset/RoI/train/RoI.txt`
`val_path: dataset/MT_dataset/RoI/val/RoI.txt`,`num_classes: 2`.

3.2 模型验证：
模型训练完毕后，可以进行验证检测AP或分割mIoU等指标：

检测：
```bash
python3 tools/det/eval.py -c configs/det/hrnet/faster_rcnn_hrnetv2p_w18_3x_defect.yml -o weights=./output/faster_rcnn_hrnetv2p_w18_3x_defect/model_final.pdparams
```

RoI分割：
```bash
python3 tools/seg/val.py --config configs/seg/ocrnet/ocrnet_hrnetw18_RoI_defect_256x256_40k.yml --model_path ./output/RoI/best_model/model.pdparams
```

## 4 全流程预测
1 配置文件：由于选用目标检测+RoI分割+后处理的串联解决方案，配置文件选用`./configs/end2end/e2e_det_RoI_seg.yml`，具体可参考[详细教程——准备全流程配置文件](./end2end/parse_config.md).

2 执行预测：
使用`tools/end2end/predict.py`脚本，执行如下命令，完成`./dataset/MT_dataset/images/val`路径下所有图像的预测：

```
python3 tools/end2end/predict.py --config ./configs/end2end/e2e_det_RoI_seg.yml --input ./dataset/MT_dataset/images/val --output_dir ./output_det_roi/
```

得到输出结果的json和可视化文件，保存在`./output_det_roi/`.具体可参考[详细教程——全流程预测](./end2end/predict.md#全流程预测)了解脚本输入和输出文件的内容解析。


## 5 全流程评估和调优

1 评估：
执行以下命令，得到工业过杀漏失指标，badcase可视化输出保存在`output`路径下，具体可参考[详细教程——全流程评估](./end2end/eval.md#全流程评估).

```
python3 tools/end2end/eval.py --input_path ./dataset/MT_dataset/val.json --pred_path ./output_det_roi/output.json --config ./configs/end2end/e2e_det_RoI_seg.yml --rules_eval --image_root ./
```

2 调优：
由于零件缺陷判断标准可能与缺陷位置、缺陷大小、长度等信息相关，可以在`./configs/end2end/e2e_det_RoI_seg.yml`文件中通过改变后处理参数调节过杀漏检指标，调整后，执行上面的命令重新评测。具体可参考[详细教程——全流程评估](./end2end/eval.md#后处理参数调整).

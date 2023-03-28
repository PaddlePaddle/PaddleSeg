# 数据相关工具

## 1. 标签可视化工具

### 1.1 coco格式json数据可视化

如果想检查coco格式的数据是否正确转换，可以使用以下命令进行简单的可视化，其中`json_path`是上面生成的json文件，`root_path`是图像目录，`save_path`是保存可视化图像的路径
```shell
python3 tools/dataset/visualize_coco.py --json_path ./dataset/MT_dataset/train.json --root_path ./ --save_path ./show/
```

### 1.2 分割标签可视化

如果想检查分割数据是否正确转换，可以使用以下命令进行简单可视化：
```shell
python3 tools/dataset/visualize_annotation.py --file_path ./dataset/MT_dataset/train.txt --save_path ./show/
```

## 2. 类别统计

若希望统计各个分割类别的数量，并计算各类别的损失权重，可以执行以下命令：

```shell
python3 tools/dataset/cal_class_weights.py --anno_path ./dataset/MT_dataset/annos/val --temperature 0.8 --num_workers 4
```
输出如下：

```
CalClassWeights INFO: Overall Class Frequency Matrix:
+-----------+------+------+-------+--------+---------+----------+
|  ClassID  |  1   |  3   |   2   |   4    |    5    |    0     |
+-----------+------+------+-------+--------+---------+----------+
| Frequency | 4561 | 8074 | 93527 | 249020 | 1006766 | 46684328 |
|  Weights  | 0.19 | 0.19 |  0.19 |  0.19  |   0.19  |   0.06   |
+-----------+------+------+-------+--------+---------+----------+
```

其中anno_path是要输入的标签图像存放路径，temperature越大，权重越平滑。

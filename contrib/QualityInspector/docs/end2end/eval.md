# 全流程指标评估

模型的检测AP, 分割mIoU指标往往无法直观地评估一套工业质检解决方案，为了能够更好在项目中落地，我们提供了工业质检行业常用的过杀漏检指标的评估方法，包含过杀指标，图像级别漏失指标，以及实例缺陷级别的漏检指标。

同时，为了进一步分析优化指标，提供了对badcase的分析，并且支持后处理NG/OK判断的参数调整重新评测，无需使用检测/分割模型重新预测结果。

## 评估方法
执行以下命令对`./dataset/MT_dataset/val.json`中的每一张图像进行评估，`--pred_path`是通过执行全流程预测得到的预测结果json文件：

```
python3 tools/end2end/eval.py --input_path ./dataset/MT_dataset/val.json --pred_path ./output_det/output.json --config ./configs/end2end/e2e_det.yml --rules_eval --image_root ./
```

具体参数说明如下：

| 参数名              |       含义                           | 默认值     |
| ------------------ | ------------------------------------| --------- |
| `--input_path`     |  带有GT框的coco格式json文件            |           |
| `--pred_path`      |  通过全流程预测得到的预测json文件        |           |
| `--config`         |  全流程预测配置文件                     |          |
| `--image_root`     |  图像保存的根目录                      |    ''     |
| `--rules_eval`     |  是否重新进行后处理判断                 |    False  |
| `--instance_level` |  是否评测实力级别漏检指标                |    True  |
| `--iou_theshold`   |  预测与gt框大于该阈值，视作召回          |     0.1   |
| `--badcase`        |  是否进行badcase可视化                 |    True    |
| `--output_path`    |  badcase保存路径                      | ./output/ |


## 输出结果和指标说明

**过杀指标**
```
Eval INFO: OK Evaluation Result:
+----------+--------+----------+-------+-------+-------+--------+
|    OK    |  ALL   | Blowhole | Break | Crack |  Fray | Uneven |
+----------+--------+----------+-------+-------+-------+--------+
|  Total   |  318   |   318    |  318  |  318  |  318  |  318   |
|    OK    |  267   |   316    |  306  |  318  |  318  |  281   |
|    NG    |   51   |    2     |   12  |   0   |   0   |   37   |
| Overkill | 16.04% |  0.63%   | 3.77% | 0.00% | 0.00% | 11.64% |
+----------+--------+----------+-------+-------+-------+--------+
```

其中，Total是OK图像总数，即OK与NG行的纵向和，`Overkill=NG/Total`，每一个类别下的NG数是指包含该类别的NG预测box或者mask的图像数量。

**图像级别漏检指标**
```
Eval INFO: Result of Image-Level NG NG Evaluation:
+--------------+-------+-----+----+--------+
| Image Level  | Total |  NG | OK | Escape |
+--------------+-------+-----+----+--------+
| Lucky Result |  132  | 118 | 14 | 10.61% |
+--------------+-------+-----+----+--------+
```

其中，Total是NG图像总数, NG表示该图像存在任意一个预测为NG的box/mask（不关注这个NG预测是否是正确的位置或类别，可称作Luckcy Recall），OK表示图像没有任何的NG预测box/mask。`Escape=OK/Total`

**实例级别漏检指标**
```
Eval INFO: Result of Instance-Level NG Evaluation:
+----------+--------+----------+--------+--------+-------+--------+
|    NG    |  ALL   | Blowhole | Break  | Crack  |  Fray | Uneven |
+----------+--------+----------+--------+--------+-------+--------+
|  Total   |  151   |    39    |   41   |   24   |   13  |   34   |
|    NG    |  127   |    36    |   31   |   20   |   12  |   28   |
|    OK    |   24   |    3     |   10   |   4    |   1   |   6    |
|  Escape  | 15.89% |  7.69%   | 24.39% | 16.67% | 7.69% | 17.65% |
+----------+--------+----------+--------+--------+-------+--------+
```
其中，Total是NG的缺陷实例总数, NG表示预测正确的实例缺陷box/mask（通过与GT进行iou计算，从而判断这个NG预测是否是正确的位置），OK表示该缺陷实例没有任何的可匹配（大于iou_theshold）的NG预测box/mask。`Escape=OK/Total`


## badcase可视化输出

badcase输出保存在`output`路径下，目录结构如下：

```
    output
    |
    |--overkill            # 过杀文件夹
    |  |--Break            # 预测出的过杀类别
    |  |  |--exp1_xxx.jpg  # 可视化的过杀图像
    |  |  |--exp1_xxx.png
    |  |  ...  
    |  |--Uneven           # 预测出的过杀类别
    |  |  |--exp1_xxx.jpg
    |  |  |--exp1_xxx.png
    |  ...
    |--escape              # 漏检文件夹
    |  |--image_level      # 图像级别漏检，图像没有任何预测结果
    |  |  |--exp1_xxx.jpg
    |  |  |--exp1_xxx.png
    |  |  ...  
    |  |--instance_level   # 实例级别漏检，图像上某个/多个缺陷漏检
    |  |  |--Break         # 漏检的真实缺陷类别
    |  |  |  |--exp1_xxx.jpg
    |  |  |  |--exp1_xxx.png
    |  |  |  ...
    |  |  |--Uneven         # 漏检的真实缺陷类别
    |  |  |  |--exp1_xxx.jpg
    |  |  |  |--exp1_xxx.png
    |  |  |  ...
    |  |  ...
```


## 后处理参数调整

由于零件缺陷判断标准可能与缺陷位置、缺陷大小、长度等信息相关，为了过滤不满足NG条件的检出，降低过杀，QualityInspector支持多种后处理算子，位于`./qinspector/ops/postprocess.py`文件中，目前包括`JudgeDetByScores`以及`JudgeByLengthWidth`, `JudgeByArea`分别是通过置信度、边长和面积判断是否是缺陷的功能。通过在全流程yml文件中配置，即可使用。


在经过指标输出和badcase可视化的图像查阅后，可以在配置文件中`./configs/end2end/e2e_det.yml`调整后处理的参数，注意此时的执行命令中一定要添加`--config ./configs/end2end/e2e_det.yml`和`--rules_eval`, 才能重新通过调整参数后的后处理模块。例如：在上面展示的指标中，发现Uneven类别的过杀指标较高，因此调整配置文件中后处理模块`JudgeDetByScores`中的`score_threshold`，将第5类Uneven的阈值调高，再重新执行上述评测命令，得到过杀结果如下：

```
Eval INFO: OK Evaluation Result:
+----------+-------+----------+-------+-------+-------+--------+
|    OK    |  ALL  | Blowhole | Break | Crack |  Fray | Uneven |
+----------+-------+----------+-------+-------+-------+--------+
|  Total   |  318  |   318    |  318  |  318  |  318  |  318   |
|    OK    |  298  |   316    |  306  |  318  |  318  |  312   |
|    NG    |   20  |    2     |   12  |   0   |   0   |   6    |
| Overkill | 6.29% |  0.63%   | 3.77% | 0.00% | 0.00% | 1.89%  |
+----------+-------+----------+-------+-------+-------+--------+
```

可见，Uneven类别的过杀明显降低。当然，此时可能存在一定程度的漏失上升，用户可根据实际项目的漏失或者过杀的既定目标进行调整。

## 其他

判断图像是NG/OK的逻辑：根据输入的json文件，判断是否有标注框，有标注框即任务是NG图像。

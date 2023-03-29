# 全流程预测

除了分析模型的检测AP, 分割mIoU指标，质检需要进行全流程的零件过杀/漏检分析，通过全流程预测能够得到零件的NG/OK（缺陷或正常）信息，以及预测框/分割的实例级别NG/OK（缺陷或正常）信息, 保存完整的预测结果，并对齐进行可视化，从而进行分析和指标优化。

## 预测用法
使用`tools/end2end/predict.py`脚本，执行如下命令，完成预测：

```
python3 tools/end2end/predict.py --config ./configs/end2end/e2e_det.yml --input ./dataset/MT_dataset/images/val --output_dir ./output_det/
```

具体参数说明如下：

| 参数名          | 含义                                 | 默认值     |
| -------------  | ------------------------------------| --------- |
| `--config`     |  全流程配置文件                       |           |
| `--input`      |  待预测的图像路径 或单个图像文件         |           |
| `--output_dir` |  保存预测文件和可视化结果路径            |`./output/`|


## 输出结果解析

若将`./configs/end2end/e2e_det.yml`中的`save`和`visualize`设置为`True`, 则会得到保存的预测结果json文件和可视化图像。

```
Pipeline INFO: Save prediction to ./output_det/output.json
Pipeline INFO: Visualize prediction to ./output_det/show/
```

执行上述命令以后，在`./output_det/`路径中会得到`./output_det/output.json`文件，在`./output_det/show/`下可得到可视化的图像。

输出的json文件保存了以每一张输入图像路径为key, 预测结果为value的dict结构，详细说明如下：

```
{
    "dataset/MT_dataset/images/val/exp6_num_127730.png": {
        "isNG": 0,  # 图像级别缺陷判断，是缺陷图1，非缺陷图0
        "pred": [], # list 保存缺陷实例级别预测结果
    }，
    "dataset/MT_dataset/images/val/exp5_num_155415.jpg": {
        "isNG": 1,  # 是缺陷图
        "pred":     # 预测出以下两个缺陷实例
        [  
            {
                "category_id": 2,  # 缺陷类别id
                "category_name": "Break", # 缺陷名
                "bbox":      #缺陷预测框x, y, w, h
                [
                    0.0,
                    6.75,
                    190.86,
                    78.03
                ],
                "score": 0.28, # 预测框得分
                "isNG": 1 # 该预测实例是缺陷
            },
            {
                "category_id": 5,
                "category_name": "Uneven",
                "bbox":
                [
                    0.0,
                    156.82,
                    195.0,
                    154.17
                ],
                "score": 0.01,
                "isNG": 0  # score为0.01,经过后处理过滤，该预测框不是缺陷
            }
        ],

  },
}
```

注意：若采用分割或者检测+RoI分割的全流程配置，则输出的结果中会增加`polygon`和`area`字段。

## 输出结果可视化：

可视化图像中包含box, polygon(若使用分割)，类别，置信度，是否是NG信息。

以Magnetic-tile-defect-datasets检测+RoI分割结果为例：

## <img src="https://github.com/Sunting78/images/blob/master/show.png" width="200"/>

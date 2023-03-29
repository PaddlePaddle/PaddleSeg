# 准备全流程预测配置文件

全流程配置文件按照模块化进行定义，包含环境，全流程，检测，分割，RoI分割以及后处理等模块信息。

配置文件保存在`./configs/end2end/`目录下, 提供了三种常用的工业质检PPL，分别是：

* [检测+后处理](../../configs/end2end/e2e_det.yml): `e2e_det.yml`
* [分割+后处理](../../configs/end2end/e2e_seg.yml): `e2e_seg.yml`
* [检测+RoI分割+后处理](../../configs/end2end/e2e_det_RoI_seg.yml): `e2e_det_RoI_seg.yml`

## 详细解读

基于检测+RoI分割+后处理的配置文件`./configs/end2end/e2e_det_RoI_seg.yml`进行详细的参数解读：

```
ENV:
  device: gpu # 运行环境
  output_dir: ./output/  # 保存输出的预测json文件和可视化图像的路径
  save: True  # 保存json文件
  visualize: False # 是否进行可视化
PipeLine:
  - Detection: # 检测模块
      config_path: ./configs/det/hrnet/faster_rcnn_hrnetv2p_w18_3x_defect.yml  # 检测算法配置文件路径
      model_path: ./output/faster_rcnn_hrnetv2p_w18_3x_defect/model_final.pdparams # 检测算法训练保存的模型路径
      score_threshold: 0.01 # 只输出置信度大于0.01的bbox
  - CropSegmentation: # 区域分割模块
      pad_scale: 0.5 #根据检测box剪裁RoI区域时变长扩大的倍数
      config_path: ./configs/seg/ocrnet/ocrnet_hrnetw18_RoI_defect_256x256_40k.yml # RoI分割训练配置文件路径
      model_path: ./output/seg_roi/best_model/model.pdparams # 分割训练保存的模型
      aug_pred : False # 是否需要多尺度+Flip推理
  - PostProcess: #后处理模块
      - JudgeDetByScores:  # 置信度判断
          score_threshold:  # 不同类别的置信度阈值，若各个类别阈值一致，则可写为score_threshold: 0.1
            1: 0.01  # 类别为1置信度小于0.01的框，判为OK框，非缺陷
            2: 0.4   # 类别为1置信度小于0.01的框，判为OK框，非缺陷
      - JudgeByLengthWidth:  # 边长判断
          len_thresh: 2   # 长或宽小于2格像素的预测，判为OK框，非缺陷
      - JudgeByArea: # 面积判断
          area_thresh:
            1: 10 #类别为1且分割像素数小于10的预测，判为OK，非缺陷
            2: 5
```

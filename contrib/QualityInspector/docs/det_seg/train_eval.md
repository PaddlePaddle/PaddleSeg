# 检测分割算法的训练

以Magnetic-tile-defect-datasets为例，在前面的[数据准备](../data/prepare_data.md)中，可以获得检测，分割，RoI分割任务对应的符合训练格式的数据，检测分割任务的配置文件分别存放在`./configs/det/`和`./configs/seg/`路径下。接下来进行不同任务的训练和评测：

## 训练
* Detection

  ```bash
    python3 tools/det/train.py -c configs/det/hrnet/faster_rcnn_hrnetv2p_w18_3x_defect.yml --use_vdl=true --vdl_log_dir=./vdl_dir/scalar --eval
  ```

* Segmentation

  ```bash
    python3 tools/seg/train.py --config configs/seg/ocrnet/ocrnet_hrnetw18_defect_256x256_40k.yml --do_eval  --use_vdl --save_interval 1000 --save_dir ./output/ --num_workers 4
  ```

## 验证
* Detection

  ```bash
    python3 tools/det/eval.py -c configs/det/hrnet/faster_rcnn_hrnetv2p_w18_3x_defect.yml -o weights=./output/faster_rcnn_hrnetv2p_w18_3x_defect/model_final.pdparams
  ```

* Segmentation

  ```bash
    python3 tools/seg/val.py --config configs/seg/ocrnet/ocrnet_hrnetw18_defect_256x256_40k.yml --model_path ./output/best_model/model.pdparams
  ```

## 其他说明
 * 为了简化，Industrial Inspection只保留了部分算法配置文件，但实际上可以使用PaddleDetection/PaddleSeg中集成的任意算法，只需将算法的config文件放到`./configs/det/`或`./configs/seg/`中。
 * 具体可参考[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/docs/tutorials/QUICK_STARTED_cn.md)和[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/docs/train/train.md)的训练和验证说明文档。
 * 未来，Industrial Inspection也将针对工业质检的难点，二次开发基于检测分割算法的更多功能。
 * 暂未支持端到端部署，用户目前可参考PaddleDetection和PaddleSeg进行模型部署，未来将支持端到端的全流程部署，敬请期待。

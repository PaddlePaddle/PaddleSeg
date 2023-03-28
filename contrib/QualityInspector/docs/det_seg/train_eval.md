# 检测分割的训练验证

以Magnetic-tile-defect-datasets为例，在前面的[数据准备](../data/prepare_data.md)中，可以获得检测，分割，RoI分割任务对应的符合训练格式的数据，检测分割任务的配置文件分别存放在`./configs/det/`和`./configs/seg/`路径下。接下来进行不同任务的训练和验证：

## 训练
* 检测

以`HRNet`作为骨架网络的`FasterRCNN`算法为例，执行下面的命令启动训练：

  ```bash
    python3 tools/det/train.py -c configs/det/hrnet/faster_rcnn_hrnetv2p_w18_3x_defect.yml --use_vdl=true --vdl_log_dir=./vdl_dir/scalar --eval
  ```

也可以执行以下命令进行使用多卡进行训练：

  ```bash
    CUDA_VISIBLE_DEVICES=0,1 python3 -m paddle.distributed.launch tools/det/train.py -c configs/det/hrnet/faster_rcnn_hrnetv2p_w18_3x_defect.yml --use_vdl=true --vdl_log_dir=./vdl_dir/scalar --eval
  ```

训练模型将保存在`./output/faster_rcnn_hrnetv2p_w18_3x_defect/`文件夹中。

* 分割

以`HRNet`作为骨架网络的`OCRNet`算法为例，执行下面的命令启动训练：

  ```bash
    python3 tools/seg/train.py --config configs/seg/ocrnet/ocrnet_hrnetw18_defect_256x256_40k.yml --do_eval  --use_vdl --save_interval 1000 --save_dir ./output/
  ```
也可以执行以下命令进行使用多卡进行训练：

  ```bash
    CUDA_VISIBLE_DEVICES=0,1 python3 -m paddle.distributed.launch --config configs/seg/ocrnet/ocrnet_hrnetw18_defect_256x256_40k.yml --do_eval  --use_vdl --save_interval 1000 --save_dir ./output/
  ```

训练模型将保存在`./output/`文件夹中。

## 验证
* 检测

使用训练过程中保存的`model_final.pdparams`对验证集进行评估, -c表示指定使用哪个配置文件 -o表示指定配置文件中的全局变量（覆盖配置文件中的设置), 注意目前只支持单卡评估。

  ```bash
    python3 tools/det/eval.py -c configs/det/hrnet/faster_rcnn_hrnetv2p_w18_3x_defect.yml -o weights=./output/faster_rcnn_hrnetv2p_w18_3x_defect/model_final.pdparams
  ```

* 分割

使用训练过程中保存的`best_model/model.pdparams`对验证集进行评估:

  ```bash
    python3 tools/seg/val.py --config configs/seg/ocrnet/ocrnet_hrnetw18_defect_256x256_40k.yml --model_path ./output/best_model/model.pdparams
  ```


## 其他说明
 * 为了简化，Industrial Inspection只保留了部分算法配置文件，但实际上可以使用PaddleDetection/PaddleSeg中集成的任意算法，只需将算法的config文件放到`./configs/det/`或`./configs/seg/`中。
 * 具体可参考[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/docs/tutorials/QUICK_STARTED_cn.md)和[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/docs/train/train.md)的训练和验证说明文档。
 * 未来，Industrial Inspection也将针对工业质检的难点，二次开发基于检测分割算法的更多功能。
 * 暂未支持端到端部署，用户目前可参考PaddleDetection和PaddleSeg进行模型部署，未来将支持端到端的全流程部署，敬请期待。

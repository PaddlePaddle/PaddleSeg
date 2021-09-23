English|[简体中文](train_cn.md)
# Model Training

## 1、Start Training 

We can train the model through the script provided by PaddleSeg. Here we use `BiseNet` model and `optic_disc` dataset to show the training process. Please make sure that you have already installed PaddleSeg, and it is located in the PaddleSeg directory. Then execute the following script:


```shell
export CUDA_VISIBLE_DEVICES=0 # Set 1 usable card
# If you are using windows, please excute following script:
# set CUDA_VISIBLE_DEVICES=0
python train.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output
```

### Parameters

| Parameter     | Effection                               | Is Required | Default           |
| :------------------ | :----------------------------------------------------------- | :--------- | :--------------- |
| iters               | Number of training iterations                                                 | No         | The value specified in the configuration file.| |
| batch_size          | Batch size on a single card                                            | No         | The value specified in the configuration file.| |
| learning_rate       | Initial learning rate                                                   | No        | The value specified in the configuration file.| |
| config              | Configuration files                                                     | Yes         | -                |
| save_dir            | The root path for saving model and visualdl log files                           | No         | output           |
| num_workers         | The number of processes used to read data asynchronously, when it is greater than or equal to 1, the child process is started to read dat  | No  | 0 |
| use_vdl             | Whether to enable visualdl to record training data                                 | No         | No               |
| save_interval       | Number of steps between model saving                                           | No         | 1000             |
| do_eval             | Whether to start the evaluation when saving the model, the best model will be saved to best_model according to mIoU at startup | No   | No  |
| log_iters           | Interval steps for printing log                                           | No         | 10               |
| resume_model        | Restore the training model path, such as: `output/iter_1000`                    | No        | None             |
| keep_checkpoint_max | Number of latest models saved                                            | No        | 5                |


## 2、Multi-card training
If you want to use multi-card training, you need to specify the environment variable `CUDA_VISIBLE_DEVICES` as `multi-card` (if not specified, all GPUs will be used by default), and use `paddle.distributed.launch` to start the training script (Can not use multi-card training under Windows, because it doesn't support nccl):

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3 # Set 4 usable cards
python -m paddle.distributed.launch train.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output
```

## 3、Resume Training：
```shell
python train.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --resume_model output/iter_500 \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output
```

## 4、Visualize Training Process

PaddleSeg will write the data during the training process into the VisualDL file, and view the log during the training process in real time. The recorded data includes:
1. Loss change trend.
2. Changes in learning rate.
3. Training time.
4. Data reading time.
5. Mean IoU trend (takes effect when the `do_eval` switch is turned on).
6. Trend of mean pixel Accuracy (takes effect when the `do_eval` switch is turned on).

Run the following command to start VisualDL to view the log
```shell
# The following command will start a service on 127.0.0.1, which supports viewing through the front-end web page. You can specify the actual ip address through the --host parameter
visualdl --logdir output/
```

Enter the suggested URL in the browser, the effect is as follows:
![](../images/quick_start_vdl.jpg)

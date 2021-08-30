# Rethinking BiSeNet For Real-time Semantic Segmentation(CVPR2021)”的STDCSeg的Paddle实现版本

## Reference

> Fan, Mingyuan, et al. "Rethinking BiSeNet For Real-time Semantic Segmentation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.

**论文:** [Rethinking BiSeNet For Real-time Semantic Segmentation](https://arxiv.org/abs/2104.13188)

## Performance

### CityScapes

| Model                  |Input Resolution | mIOU |
| -----------------------|---| -------- |
| STDC2-Seg50 (Pytorch)   | 1024x512 | 74.2     |
| STDC2-Seg50 (Paddle)| 1024x512| 74.62  |

## Model info

**NOTE**：配置文件中需要指定模型类型和预训练backbone参数路径

  backbone_type(str): 'STDCNet1446' 表示STDC2 或者 'STDCNet813' 表示STDC1
  
  pretrained_backbone(str): 'pretrained/STDCNet1446_76.47.pdiparams' # 自定义预训练backbone路径
  
  num_classes(int,optional): 目标类别数

  pretrained (str, optional): 预训练模型路径
 
**关于模型的其他信息**：

采用80000 iter，batch_size=36，base_lr=0.01 warmup+poly的学习率策略（目前paddleseg暂时不支持直接配置调用，TODO），**STDCNet2-Seg50模型在Cityscaps VAL数据集上达到了74.62的mIOU**

| 信息 | 说明 |
| --- | --- |
| pretrained_backbone | [pretrained_backbone: 提取码：tss7](https://pan.baidu.com/s/16kh3aHTBBX6wfKiIG-y3yA)|
| STDC2-Seg50(model+log) | [STDC2-Seg50(model+log): 提取码：nchx](https://pan.baidu.com/s/1sFHqZWhcl8hFzGCrXu_c7Q)|
| 原始Github项目 | [STDCNet-Paddle](https://github.com/CuberrChen/STDCNet-Paddle/tree/master) |
| 在线运行项目 | [AIStudio notebook](https://aistudio.baidu.com/aistudio/projectdetail/2206098) |


Refrence Code:
- [Paper's PyTorch implementation](https://github.com/MichaelFan01/STDC-Seg)

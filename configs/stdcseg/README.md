# Rethinking BiSeNet For Real-time Semantic Segmentation

## Reference

> Fan, Mingyuan, et al. "Rethinking BiSeNet For Real-time Semantic Segmentation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.


## Performance

### CityScapes

| Model      |Resolution | Training Iters | mIOU |
| -------------|---------|---------------|------ |
| STDC2-Seg50 (Paddle)| 1024x512|80k| 74.62  |

## Model info

**NOTE**：配置文件中需要指定模型类型和预训练backbone参数路径

  backbone: STDCNet1446 表示STDC2 或者 STDCNet813 表示STDC1
    type: STDCNet1446
    pretrained: 'backbone参数路径/STDCNet1446_76.47.pdiparams' 

**关于模型的其他信息**：

采用80000 iter，batch_size=36，base_lr=0.01 warmup+poly的学习率策略，**STDCNet2-Seg50模型在Cityscaps VAL数据集上达到了74.62的mIOU**

| 信息 | 说明 |
| --- | --- |
| pretrained_backbone | [pretrained_backbone: 提取码：tss7](https://pan.baidu.com/s/16kh3aHTBBX6wfKiIG-y3yA)|
| STDC2-Seg50(model+log) | [STDC2-Seg50(model+log): 提取码：nchx](https://pan.baidu.com/s/1sFHqZWhcl8hFzGCrXu_c7Q)|


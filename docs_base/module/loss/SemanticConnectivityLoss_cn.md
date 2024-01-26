简体中文 | [English](SemanticConnectivityLoss_en.md)
## [SemanticConnectivityLoss](../../../paddleseg/models/losses/semantic_connectivity_loss.py)
SCL（Semantic Connectivity-aware Learning）框架，它引入了SC Loss (Semantic Connectivity-aware Loss)，从连通性的角度提升分割结果的质量。支持多类别分割。

论文信息：
    Lutao Chu, Yi Liu, Zewu Wu, Shiyu Tang, Guowei Chen, Yuying Hao, Juncai Peng, Zhiliang Yu, Zeyu Chen, Baohua Lai, Haoyi Xiong.
    "PP-HumanSeg: Connectivity-Aware Portrait Segmentation with a Large-Scale Teleconferencing Video Dataset"
    In WACV 2022 workshop
    https://arxiv.org/abs/2112.07146

执行步骤：
步骤1，连通域计算
步骤2，连通域匹配与SC Loss计算
```python
class paddleseg.models.losses.SemanticConnectivityLoss(
            ignore_index = 255,
            max_pred_num_conn = 10,
            use_argmax = True
)
```

## 语义连通性学习(SCL) 使用指南

### 参数
* **ignore_index** (int): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。当标注图中存在无法标注（或很难标注）的像素时，可以将其标注为某特定灰度值。在计算损失值时，其与原图像对应位置的像素将不作为损失函数的自变量。 *默认:``255``*
* **max_pred_num_conn** (int): 预测连通域的最大数量。在训练开始时，往往存在大量连通域，导致计算非常耗时。因此，有必要限制预测连通域的最大数量，超出最大数量的连通域将不参与计算。
* **use_argmax** (bool): 是否对logits进行argmax操作。

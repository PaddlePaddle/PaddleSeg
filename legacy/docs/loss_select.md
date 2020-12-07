# Loss选择

目前PaddleSeg提供了6种损失函数，分别为
- Softmax loss (softmax with cross entropy loss)
- Weighted softmax loss (weighted softmax with cross entropy loss)
- Dice loss (dice coefficient loss)
- Bce loss (binary cross entropy loss)
- Lovasz hinge loss
- Lovasz softmax loss

## 类别不均衡问题
在图像分割任务中，经常出现类别分布不均匀的情况，例如：工业产品的瑕疵检测、道路提取及病变区域提取等。

针对这个问题，您可使用Weighted softmax loss、Dice loss、Lovasz hinge loss和Lovasz softmax loss进行解决。

### Weighted softmax loss
Weighted softmax loss是按类别设置不同权重的softmax loss。

通过设置`cfg.SOLVER.CROSS_ENTROPY_WEIGHT`参数进行使用。  
默认为None. 如果设置为'dynamic'，会根据每个batch中各个类别的数目，动态调整类别权重。
也可以设置一个静态权重(list的方式)，比如有3类，每个类别权重可以设置为[0.1, 2.0, 0.9]. 示例如下
```yaml
SOLVER:
    CROSS_ENTROPY_WEIGHT: 'dynamic'
```

### Dice loss
参见[Dice loss教程](./dice_loss.md)

### Lovasz hinge loss和Lovasz softmax loss
参见[Lovasz loss教程](./lovasz_loss.md)

# loss的选择

在PaddleSeg中，目前支持`softmax_loss(sotfmax with cross entroy loss)`, 
`dice_loss(dice coefficient loss)`, `bce_loss(binary cross entroy loss)`三种损失函数。
根据数据集的情况选择合适的损失函数能够明显的改善分割结果。例如对于DeepGlobe Road Extraction数据集，
道路占比仅为4.5%，类别严重不平衡，这时候使用softmax_loss，背景将会占据主导地位，使得模型偏向于背景。
而dice_loss通过计算预测与标注之间的重叠部分计算损失函数，避免了类别不均衡带来的影响，能够取得更好的效果。
在实际应用中dice loss往往与bce loss结合使用，提高模型训练的稳定性。
DeepGlobe Road Extraction的训练集中，随机选取800张图片作为训练数据，选取200张作为评估数据，对softmax_loss
和dice_loss + bce loss进行实验比较，如下图所示:
<p align="center">
  <img src="./imgs/loss_comparison.png" hspace='10' height="208" width="516"/> <br />
 </p>
图中橙色曲线为dice_loss + bce loss，最高mIoU为76.02%，蓝色曲线为softmax_loss， 最高mIoU为73.62%。

## loss的指定
通过cfg.SOLVER.LOSS参数可以选择训练时的损失函数，目前支持`softmax_loss(sotfmax with cross entroy loss)`, 
`dice_loss(dice coefficient loss)`, `bce_loss(binary cross entroy loss)`三种损失函数。
其中`dice_loss`和`bce_loss`仅在两类分割问题中适用，`softmax_loss`不能与`dice_loss`
或`bce_loss`组合，`dice_loss`可以和`bce_loss`组合使用。使用示例如下：

`['softmax_loss']`或`['dice_loss','bce_loss']`

## loss的定义

* softmax_loss

多分类交叉熵损失函数，公式如下所示：

![equation](http://latex.codecogs.com/gif.latex?softmax\\_loss=\sum_{i=1}^Ny_i{log(p_i)}) 

<br/>

* dice_loss

dice loss是对分割评价指标优化的损失函数，是一种二分类的损失函数，在前景背景比例严重不平衡的情况下往往能取到较好的效果。
在实际应用中dice loss往往与bce loss结合使用，提高模型训练的稳定性

![equation](http://latex.codecogs.com/gif.latex?dice\\_loss=1-\frac{2|Y\bigcap{P}|}{|Y|+|P|}) 

其中 ![equation](http://latex.codecogs.com/gif.latex?|Y\bigcap{P}|) 表示*Y*和*P*的共有元素数，
实际计算通过求两者的乘积之和进行计算。如下所示：

<p align="center">
  <img src="./imgs/dice1.png" hspace='10' height="68" width="513"/> <br />
 </p>

[dice系数](https://zh.wikipedia.org/wiki/Dice%E7%B3%BB%E6%95%B0)

<br/>
<br/>

* bce_loss

二分类用的交叉熵损失函数，公式如下所示：

![equation](http://latex.codecogs.com/gif.latex?bce\\_loss=y_i{log(p_i)}+(1-y_i)log(1-p_i))

其中![equation](http://latex.codecogs.com/gif.latex?y_i)和*Y*为标签，
 ![equation](http://latex.codecogs.com/gif.latex?p_i)和*P*为预测结果

## 默认值

['softmax_loss']

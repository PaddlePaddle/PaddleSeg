简体中文|[English](evaluate.md)
## 模型评估

### 1.**配置化驱动**方式下评估和预测

#### 评估

训练完成后，用户可以使用评估脚本val.py来评估模型效果。假设训练过程中迭代次数（iters）为1000，保存模型的间隔为500，即每迭代1000次数据集保存2次训练模型。因此一共会产生2个定期保存的模型，加上保存的最佳模型`best_model`，一共有3个模型，可以通过`model_path`指定期望评估的模型文件。

```
!python val.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --model_path output/iter_1000/model.pdparams
```

如果想进行多尺度翻转评估，可通过传入`--aug_eval`进行开启，然后通过`--scales`传入尺度信息， `--flip_horizontal`开启水平翻转， `flip_vertical`开启垂直翻转。使用示例如下：

```
python val.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --model_path output/iter_1000/model.pdparams \
       --aug_eval \
       --scales 0.75 1.0 1.25 \
       --flip_horizontal
```

如果想进行滑窗评估，可通过传入`--is_slide`进行开启， 通过`--crop_size`传入窗口大小， `--stride`传入步长。使用示例如下：

```
python val.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --model_path output/iter_1000/model.pdparams \
       --is_slide \
       --crop_size 256 256 \
       --stride 128 128
```

在图像分割领域中，评估模型质量主要是通过三个指标进行判断，准确率（acc）、平均交并比（Mean Intersection over Union，简称mIoU）、Kappa系数。

- 准确率：指类别预测正确的像素占总像素的比例，准确率越高模型质量越好。
- 平均交并比：对每个类别数据集单独进行推理计算，计算出的预测区域和实际区域交集除以预测区域和实际区域的并集，然后将所有类别得到的结果取平均。在本例中，正常情况下模型在验证集上的mIoU指标值会达到0.80以上，显示信息示例如下所示，第3行的**mIoU=0.8526**即为mIoU。
- Kappa系数：一个用于一致性检验的指标，可以用于衡量分类的效果。kappa系数的计算是基于混淆矩阵的，取值为-1到1之间，通常大于0。其公式如下所示，P0P_0*P*0为分类器的准确率，PeP_e*P**e*为随机分类器的准确率。Kappa系数越高模型质量越好。

<a href="https://www.codecogs.com/eqnedit.php?latex=Kappa=&space;\frac{P_0-P_e}{1-P_e}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Kappa=&space;\frac{P_0-P_e}{1-P_e}" title="Kappa= \frac{P_0-P_e}{1-P_e}" /></a>

随着评估脚本的运行，最终打印的评估日志如下。

```
...
2021-01-13 16:41:29 [INFO]	Start evaluating (total_samples=76, total_iters=76)...
76/76 [==============================] - 2s 30ms/step - batch_cost: 0.0268 - reader cost: 1.7656e-
2021-01-13 16:41:31 [INFO]	[EVAL] #Images=76 mIoU=0.8526 Acc=0.9942 Kappa=0.8283
2021-01-13 16:41:31 [INFO]	[EVAL] Class IoU:
[0.9941 0.7112]
2021-01-13 16:41:31 [INFO]	[EVAL] Class Acc:
[0.9959 0.8886]
```

### 2.**API**方式下评估和预测

#### 评估

构建模型
```
from paddleseg.models import BiSeNetV2
model = BiSeNetV2(num_classes=2,
                 lambd=0.25,
                 align_corners=False,
                 pretrained=None)
```

加载模型参数

```
model_path = 'output/best_model/model.pdparams'#最优模型路径
if model_path:
    para_state_dict = paddle.load(model_path)  
    model.set_dict(para_state_dict)            #加载模型参数
    print('Loaded trained params of model successfully')
else:
    raise ValueError('The model_path is wrong: {}'.format(model_path))
```

构建验证集

```
# 构建验证用的transforms
import paddleseg.transforms as T
transforms = [
    T.Resize(target_size=(512, 512)),
    T.Normalize()
]

# 构建验证集
from paddleseg.datasets import OpticDiscSeg
val_dataset = OpticDiscSeg(
    dataset_root='data/optic_disc_seg',
    transforms=transforms,
    mode='val'
)
```

**Evaluate** API 参数解析

```
paddleseg.core.evaluate(
                        model,
                        eval_dataset,
                        aug_eval=False,
                        scales=1.0,  
                        flip_horizontal=True,
                        flip_vertical=False,
                        is_slide=False,
                        stride=None,
                        crop_size=None,
                        num_workers=0  
)
```

- 参数说明如下

| 参数名          | 数据类型          | 用途                                                 | 是否必选项 | 默认值 |
| --------------- | ----------------- | ---------------------------------------------------- | ---------- | ------ |
| model           | nn.Layer          | 分割模型                                             | 是         | -      |
| eval_dataset    | paddle.io.Dataset | 验证集DataSet                                        | 是         | -      |
| aug_eval        | bool              | 是否使用数据增强                                     | 否         | False  |
| scales          | list/float        | 多尺度评估，aug_eval为True时生效                     | 否         | 1.0    |
| flip_horizontal | bool              | 是否使用水平翻转，aug_eval为True时生效               | 否         | True   |
| flip_vertical   | bool              | 是否使用垂直翻转，aug_eval为True时生效               | 否         | False  |
| is_slide        | bool              | 是否通过滑动窗口进行评估                             | 否         | False  |
| stride          | tuple/list        | 设置滑动窗宽的宽度和高度，is_slide为True时生效       | 否         | None   |
| crop_size       | tuple/list        | 设置滑动窗口的裁剪的宽度和高度，is_slide为True时生效 | 否         | None   |
| num_workers     | int               | 多线程数据加载                                       | 否         | 0      |

**注意** 如果你想提升显存利用率，可以适当的提高 num_workers 的设置，以防GPU工作期间空等。


导入API接口，开始评估

```
from paddleseg.core import evaluate
evaluate(
        model,
        val_dataset #paddle.io.Dataset，验证集DataSet
)
```

多尺度+翻转评估

```
evaluate(
        model,
        val_dataset,
        aug_eval=True,  #是否使用数据增强
        scales=[0.75, 1.0, 1.25],  #缩放因子
        flip_horizontal=True)  #是否水平翻转
```

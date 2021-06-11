# cfg

BASIC Group存放所有通用配置

## `BATCH_SIZE`

训练、评估、可视化时所用的BATCH大小

### 默认值

1（需要根据实际需求填写）

### 注意事项

* 当指定了多卡运行时，PaddleSeg会将数据平分到每张卡上运行，因此每张卡单次运行的数量为 BATCH_SIZE // dev_count

* 多卡运行时，请确保BATCH_SIZE可被dev_count整除

* 增大BATCH_SIZE有利于模型训练时的收敛速度，但是会带来显存的开销。请根据实际情况评估后填写合适的值

* 目前PaddleSeg提供的很多预训练模型都有BN层，如果BATCH SIZE设置为1，则此时训练可能不稳定导致nan

<br/>
<br/>

## `TRAIN_CROP_SIZE`

训练时所对图片裁剪的大小（格式为 *[宽, 高]* ）

### 默认值

无（需要用户自己填写）

### 注意事项
`TRAIN_CROP_SIZE`可以设置任意大小，具体如何设置根据数据集而定。

<br/>
<br/>

## `EVAL_CROP_SIZE`

评估时所对图片裁剪的大小（格式为 *[宽, 高]* ）

### 默认值

无（需要用户自己填写）

### 注意事项
`EVAL_CROP_SIZE`的设置需要满足以下条件，共有3种情形：
- 当`AUG.AUG_METHOD`为unpadding时，`EVAL_CROP_SIZE`的宽高应不小于`AUG.FIX_RESIZE_SIZE`的宽高。
- 当`AUG.AUG_METHOD`为stepscaling时，`EVAL_CROP_SIZE`的宽高应不小于原图中最长的宽高。
- 当`AUG.AUG_METHOD`为rangescaling时，`EVAL_CROP_SIZE`的宽高应不小于缩放后图像中最长的宽高。

<br/>
<br/>

# cfg.DATALOADER

DATALOADER Group存放所有与数据加载相关的配置

## `NUM_WORKERS`

数据载入时的并发数量

### 默认值

8

### 注意事项

* 该选项只在`pdseg/train.py`和`pdseg/eval.py`中使用到
* 该字段表示数据预处理时的进程数量，只有在`pdseg/train.py`或者`pdseg/eval.py`中打开了`--use_mpio`开关有效。一般该字段使用默认值即可

<br/>
<br/>

## `BUF_SIZE`

数据载入时的缓存队列大小

### 默认值

256

<br/>
<br/>

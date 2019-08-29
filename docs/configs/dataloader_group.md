# cfg.DATALOADER

DATALOADER Group存放所有与数据加载相关的配置

## `NUM_WORKERS`

数据载入时的并发数量

### 默认值

8

### 注意事项

* 该选项只在`pdseg/train.py`和`pdseg/eval.py`中使用到
* 当使用多线程时，该字段表示线程数量，使用多进程时，该字段表示进程数量。一般该字段使用默认值即可

<br/>
<br/>

## `BUF_SIZE`

数据载入时的缓存队列大小

### 默认值

256

<br/>
<br/>

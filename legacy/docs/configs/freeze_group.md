# cfg.FREEZE

FREEZE Group存放所有与模型导出相关的配置

## `MODEL_FILENAME`

导出模型后所保存的模型文件名

### 默认值

`__model__`

### 注意事项
* 仅在使用`pdseg/export_model.py` 脚本导出模型时，该字段必填

<br/>
<br/>

## `PARAMS_FILENAME`

导出模型后所保存的参数文件名

### 默认值

`__params__`

### 注意事项
* 仅在使用`pdseg/export_model.py` 脚本导出模型时，该字段必填

<br/>
<br/>

## `SAVE_DIR`

保存导出模型的主目录

### 默认值

`freeze_model`

### 注意事项
* 仅在使用`pdseg/export_model.py` 脚本导出模型时，该字段必填

<br/>
<br/>

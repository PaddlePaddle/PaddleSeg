# cfg.MODEL.HRNET

MODEL.HRNET 子Group存放所有和HRNet模型相关的配置

## `STAGE2.NUM_MODULES`

HRNet在第二阶段执行modularized block（multi-resolution parallel convolution + multi-resolution fusion)的重复次数

### 默认值

1

<br/>
<br/>

## `STAGE2.NUM_CHANNELS`

HRNet在第二阶段各个分支的通道数

### 默认值

[40, 80]

<br/>
<br/>

## `STAGE3.NUM_MODULES`

HRNet在第三阶段执行modularized block的重复次数

### 默认值

4

<br/>
<br/>

## `STAGE3.NUM_CHANNELS`

HRNet在第三阶段各个分支的通道数

### 默认值

[40, 80, 160]

<br/>
<br/>

## `STAGE4.NUM_MODULES`

HRNet在第四阶段执行modularized block的重复次数

### 默认值

3

<br/>
<br/>

## `STAGE4.NUM_CHANNELS`

HRNet在第四阶段各个分支的通道数

### 默认值

[40, 80, 160, 320]

<br/>
<br/>
# ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network

## Reference

> Sachin Mehta, Mohammad Rastegari, Linda Shapiro, and Hannaneh Hajishirzi. "ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network."


## Performance

### CityScapes

| Model | Backbone | Resolution | Training Iters | mIoU | Links |
|:---:|:---:|:---:|:---:|:---:|:---:|
|ESPNetV2|ESPNetV2|1024x512|120000|69.02%|[model 提取码：sd4e](https://pan.baidu.com/s/1zNXUF2n1QSx7ayDqqGocTA)|


# 其他说明
1、paddlepaddle==2.1.2版本交叉熵损失函数有bug，请在develop版本运行；  
2、paddleseg=develop 在paddlepaddle==2.1.2交叉熵损失函数传入weight时有bug，请更换为paddleseg-release2.2下的交叉熵损失。  

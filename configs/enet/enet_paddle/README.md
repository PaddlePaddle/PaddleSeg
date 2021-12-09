# enet_paddle
* [1. 简介](#简介)
* [2. 环境准备](#环境准备)
* [3. 数据准备](#数据准备)
* [4. 训练](#训练)
* [5. 模型评价](#模型评价)
* [6. 模型导出](#模型导出)
* [7 模型推理](#模型推理)

<a name="简介"></a>

## 1. 简介

语义分割的落地（应用于移动端设备如手机、可穿戴设备等低计算能力设备）是一个很重要的问题，而最近提出的网络中，虽然有着较高的准确率，但是其实时性不强，也就是训练、推理的速度太慢，无法应用于真实的应用场景中。针对以上问题，作者Adam Paszke及其团队提出了ENet，在保证较高准确率的基础上还能保证网络更轻量，更快，适合部署在手机等可移动嵌入式设备。经过测试，ENet在CamVid, Cityscapes 和 SUN等数据集中均达到当时的SOTA。

enet_paddle是基于百度PaddlePaddle旗下端到端开发套件PaddleSeg实现的。

<a name="环境准备"></a>

## 2.环境准备
以在百度Aistudio平台下运行项目为例。

拉取PaddleSeg官方仓库

'''shell
git clone https://github.com/PaddlePaddle/PaddleSeg.git
'''

<a name="数据准备"></a>

## 3.数据准备

在百度Aistudio里创建项目，在创建项目时要挂载Cityscapes数据集。

解压数据集
'''shell
tar -xvf /home/aistudio/data/data64550/cityscapes.tar -C /home/aistudio/data
'''

<a name="训练"></a>

## 4.训练

'''bash
python PaddleSeg/train.py --num_workers 8 --config PaddleSeg/configs/enet/enet_cityscapes_1024x512_adam_0.002_80k.yml
'''

<a name="模型评价"></a>

## 5.模型评价

'''bash
python PaddleSeg/val.py --num_workers 8 --config PaddleSeg/configs/enet/enet_cityscapes_1024x512_adam_0.002_80k.yml \
--model_path=output/best_model/model.pdparams
'''

<a name="模型导出"></a>

## 6. 模型导出

'''bash
python PaddleSeg/export.py --config PaddleSeg/configs/enet/enet_cityscapes_1024x512_adam_0.002_80k.yml --model_path output/best_model/model.pdparams --save_dir output/output_model_inference --input_shape 1 3 2048 1024
'''

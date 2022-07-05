# CAE Vision Transformer for Semantic Segmentaion

This repo contains the supported code and configuration files to reproduce semantic segmentaion results of [Context Autoencoder for Self-Supervised Representation Learning](https://arxiv.org/pdf/2202.03026.pdf). 

## Updates

***06/9/2022*** Initial commits

## Results and Models

### ADE20K

| Backbone | Method | Crop Size | Lr Schd | mIoU | #params | FLOPs | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Vit-B | UperNet | 512x512 | 160K | 2e-4 | 49.69 | 81M | 1038G | [config]() | [github]()/[baidu]() | [github]()/[baidu]() |


**Notes**: 

- **Pre-trained models can be downloaded from [ CAE Vision Transformer for ImageNet Classification](https://github.com/)**.


## Usage

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md#installation) for installation and dataset preparation.

### Inference
```
# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <SEG_CHECKPOINT_FILE> <GPU_NUM> --eval mIoU

# multi-gpu, multi-scale testing
tools/dist_test.sh <CONFIG_FILE> <SEG_CHECKPOINT_FILE> <GPU_NUM> --aug-test --eval mIoU
```

### Training

To train with pre-trained models, run:
```
# multi-gpu training
python -u -m paddle.distributed.launch train.py \
--config  <CONFIG_FILE> \
--do_eval --use_vdl --save_interval <num> --save_dir output/upernet_vit_fpn \
--num_workers 4
```
For example, to train an UPerNet model with a `Vit-B` backbone and 8 gpus, run:
```
python -u -m paddle.distributed.launch train.py \
--config configs/upernet/upernet_vit_base_ade20k_512x512_160k.yml \
--do_eval --use_vdl --save_interval 8000 --save_dir output/upernet_vit_fpn \
--num_workers 4```

**Notes:** 
- The default learning rate and training schedule is for 8 GPUs and 2 imgs/gpu.


## Citing Context Autoencoder for Self-Supervised Representation Learning
```
@article{chen2022context,
  title={Context autoencoder for self-supervised representation learning},
  author={Chen, Xiaokang and Ding, Mingyu and Wang, Xiaodi and Xin, Ying and Mo, Shentong and Wang, Yunhao and Han, Shumin and Luo, Ping and Zeng, Gang and Wang, Jingdong},
  journal={arXiv preprint arXiv:2202.03026},
  year={2022}
}
```


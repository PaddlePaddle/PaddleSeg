import os
import numpy as np
from HumanSeg.datasets.dataset import Dataset
from HumanSeg.models import HumanSegMobile
from HumanSeg.transforms import transforms

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((192, 192)),
    transforms.Normalize()
])

eval_transforms = transforms.Compose(
    [transforms.Resize((192, 192)),
     transforms.Normalize()])

data_dir = '/ssd1/chenguowei01/dataset/humanseg/supervise.ly'
train_list = '/ssd1/chenguowei01/dataset/humanseg/supervise.ly/train.txt'
val_list = '/ssd1/chenguowei01/dataset/humanseg/supervise.ly/val.txt'

train_dataset = Dataset(
    data_dir=data_dir,
    file_list=train_list,
    transforms=train_transforms,
    num_workers='auto',
    buffer_size=100,
    parallel_method='thread',
    shuffle=True)

eval_dataset = Dataset(
    data_dir=data_dir,
    file_list=val_list,
    transforms=eval_transforms,
    num_workers='auto',
    buffer_size=100,
    parallel_method='thread',
    shuffle=False)

model = HumanSegMobile(num_classes=2)

model.train(
    num_epochs=2,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    save_interval_epochs=1,
    train_batch_size=128,
    pretrain_weights='output/best_model',
    log_interval_steps=2,
    save_dir='output/quant_train',
    learning_rate=0.001,
    use_vdl=True,
    quant=True)

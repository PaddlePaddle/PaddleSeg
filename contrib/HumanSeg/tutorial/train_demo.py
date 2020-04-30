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
    num_epochs=100,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    save_interval_epochs=5,
    train_batch_size=256,
    # resume_weights='/Users/chenguowei01/PycharmProjects/github/PaddleSeg/contrib/HumanSeg/output/epoch_20',
    log_interval_steps=2,
    save_dir='output',
    use_vdl=True,
)

model.evaluate(eval_dataset, batch_size=10)
im_file = '/ssd1/chenguowei01/dataset/humanseg/supervise.ly/images/8d308c9cc0326a3bdfc90f7f6e1813df89786122.jpg'
result = model.predict(im_file)
import cv2
cv2.imwrite('pred.png', result['label_map'] * 200)

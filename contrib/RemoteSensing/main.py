import sys
import os
import os.path as osp
import numpy as np
from PIL import Image as Image

#================================setting========================
# set gpu
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# batch size
batch_size = 4
# training epochs
epochs = 100

# number of data channel
channel = 10
# model save directory
save_dir = 'saved_model/snow2019_unet_all_channel_clip_norm_2'
# dataset directory
data_dir = "../../../dataset/snow2019/all_channel_data/"
#=============================================================

sys.path.append(osp.join(os.getcwd(), '..'))
import RemoteSensing.transforms.transforms as T
from RemoteSensing.readers.reader import Reader
from RemoteSensing.models import UNet, load_model

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

train_list = osp.join(data_dir, 'train.txt')
val_list = osp.join(data_dir, 'val.txt')
label_list = osp.join(data_dir, 'labels.txt')

os.system('cp ./{} {}'.format(__file__, osp.join(save_dir, __file__)))

train_transforms = T.Compose([
    T.RandomVerticalFlip(0.5),
    T.RandomHorizontalFlip(0.5),
    T.ResizeStepScaling(0.5, 2.0, 0.25),
    T.RandomPaddingCrop(769),
    T.Clip(min_val=0, max_val=3400),
    T.Normalize(max_val=3400, mean=[0.5] * channel, std=[0.5] * channel),
])

eval_transforms = T.Compose([
    T.Padding([1049, 1049]),
    T.Clip(min_val=0, max_val=3400),
    T.Normalize(max_val=3400, mean=[0.5] * channel, std=[0.5] * channel),
])

train_reader = Reader(
    data_dir=data_dir,
    file_list=train_list,
    label_list=label_list,
    transforms=train_transforms,
    num_workers=8,
    buffer_size=16,
    shuffle=True,
    parallel_method='thread')

eval_reader = Reader(
    data_dir=data_dir,
    file_list=val_list,
    label_list=label_list,
    transforms=eval_transforms,
    num_workers=8,
    buffer_size=16,
    shuffle=False,
    parallel_method='thread')

model = UNet(
    num_classes=2, input_channel=channel, use_bce_loss=True, use_dice_loss=True)

model.train(
    num_epochs=epochs,
    train_reader=train_reader,
    train_batch_size=batch_size,
    eval_reader=eval_reader,
    save_interval_epochs=5,
    log_interval_steps=10,
    save_dir=save_dir,
    pretrain_weights=None,
    optimizer=None,
    learning_rate=0.01,
)

# predict
model = load_model(osp.join(save_dir, 'best_model'))
pred_dir = osp.join(save_dir, 'pred')
if not osp.exists(pred_dir):
    os.mkdir(pred_dir)

color_map = [0, 0, 0, 255, 255, 255]

with open(val_list) as f:
    lines = f.readlines()
    for line in lines:
        img_path = line.split(' ')[0]
        print('Predicting {}'.format(img_path))
        img_path_ = osp.join(data_dir, img_path)

        pred = model.predict(img_path_)

        pred_name = osp.basename(img_path).rstrip('npy') + 'png'
        pred_path = osp.join(pred_dir, pred_name)
        pred_mask = Image.fromarray(pred.astype(np.uint8), mode='P')
        pred_mask.putpalette(color_map)
        pred_mask.save(pred_path)

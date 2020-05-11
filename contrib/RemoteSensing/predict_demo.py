import os
import os.path as osp
import numpy as np
from PIL import Image as Image
import argparse
from models import load_model


def parse_args():
    parser = argparse.ArgumentParser(description='RemoteSensing predict')
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        help='dataset directory',
        default=None,
        type=str)
    parser.add_argument(
        '--load_model_dir',
        dest='load_model_dir',
        help='model load directory',
        default=None,
        type=str)
    return parser.parse_args()


args = parse_args()

data_dir = args.data_dir
load_model_dir = args.load_model_dir

# predict
model = load_model(osp.join(load_model_dir, 'best_model'))
pred_dir = osp.join(load_model_dir, 'pred')
if not osp.exists(pred_dir):
    os.mkdir(pred_dir)

val_list = osp.join(data_dir, 'val.txt')
color_map = [0, 0, 0, 255, 255, 255]
with open(val_list) as f:
    lines = f.readlines()
    for line in lines:
        img_path = line.split(' ')[0]
        print('Predicting {}'.format(img_path))
        img_path_ = osp.join(data_dir, img_path)

        pred = model.predict(img_path_)

        # 以伪彩色png图片保存预测结果
        pred_name = osp.basename(img_path).rstrip('npy') + 'png'
        pred_path = osp.join(pred_dir, pred_name)
        pred_mask = Image.fromarray(pred.astype(np.uint8), mode='P')
        pred_mask.putpalette(color_map)
        pred_mask.save(pred_path)

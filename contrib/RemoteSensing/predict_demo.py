import os
import os.path as osp
import sys
import numpy as np
from PIL import Image as Image
import argparse
from models import load_model


def parse_args():
    parser = argparse.ArgumentParser(description='RemoteSensing predict')
    parser.add_argument(
        '--single_img',
        dest='single_img',
        help='single image path to predict',
        default=None,
        type=str)
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        help='dataset directory',
        default=None,
        type=str)
    parser.add_argument(
        '--file_list',
        dest='file_list',
        help='file name of predict file list',
        default=None,
        type=str)
    parser.add_argument(
        '--load_model_dir',
        dest='load_model_dir',
        help='model load directory',
        default=None,
        type=str)
    parser.add_argument(
        '--save_img_dir',
        dest='save_img_dir',
        help='save directory name of predict results',
        default='predict_results',
        type=str)
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


args = parse_args()
data_dir = args.data_dir
file_list = args.file_list
single_img = args.single_img
load_model_dir = args.load_model_dir
save_img_dir = args.save_img_dir
if not osp.exists(save_img_dir):
    os.makedirs(save_img_dir)

# predict
model = load_model(load_model_dir)

color_map = [0, 0, 0, 0, 255, 0]
if single_img is not None:
    pred = model.predict(single_img)
    # 以伪彩色png图片保存预测结果
    pred_name = osp.basename(single_img).rstrip('npy') + 'png'
    pred_path = osp.join(save_img_dir, pred_name)
    pred_mask = Image.fromarray(pred['label_map'].astype(np.uint8), mode='P')
    pred_mask.putpalette(color_map)
    pred_mask.save(pred_path)
elif (file_list is not None) and (data_dir is not None):
    with open(osp.join(data_dir, file_list)) as f:
        lines = f.readlines()
        for line in lines:
            img_path = line.split(' ')[0]
            print('Predicting {}'.format(img_path))
            img_path_ = osp.join(data_dir, img_path)

            pred = model.predict(img_path_)

            # 以伪彩色png图片保存预测结果
            pred_name = osp.basename(img_path).rstrip('npy') + 'png'
            pred_path = osp.join(save_img_dir, pred_name)
            pred_mask = Image.fromarray(
                pred['label_map'].astype(np.uint8), mode='P')
            pred_mask.putpalette(color_map)
            pred_mask.save(pred_path)
else:
    raise Exception(
        'You should either set the parameter single_img, or set the parameters data_dir, file_list.'
    )

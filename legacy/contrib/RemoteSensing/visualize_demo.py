import os
import os.path as osp
import argparse
from PIL import Image as Image
from models.utils import visualize as vis


def parse_args():
    parser = argparse.ArgumentParser(description='RemoteSensing visualization')
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        help='Dataset directory',
        default=None,
        type=str)
    parser.add_argument(
        '--file_list',
        dest='file_list',
        help='The name of file list that need to be visualized',
        default=None,
        type=str)
    parser.add_argument(
        '--pred_dir',
        dest='pred_dir',
        help='Directory for predict results',
        default=None,
        type=str)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='Save directory for visual results',
        default=None,
        type=str)
    return parser.parse_args()


args = parse_args()
data_dir = args.data_dir
pred_dir = args.pred_dir
save_dir = args.save_dir
file_list = osp.join(data_dir, args.file_list)
if not osp.exists(save_dir):
    os.mkdir(save_dir)

with open(file_list) as f:
    lines = f.readlines()
    for line in lines:
        img_list = []

        img_line = line.split(' ')[0]
        img_name = osp.basename(img_line).replace('data.tif', 'photo.png')
        img_path = osp.join(data_dir, 'data_vis', img_name)
        img = Image.open(img_path)
        img_list.append(img)
        print('visualizing {}'.format(img_path))

        gt_line = line.split(' ')[1].rstrip('\n')
        gt_path = osp.join(data_dir, gt_line)
        gt_pil = Image.open(gt_path)
        img_list.append(gt_pil)

        pred_name = osp.basename(img_line).replace('tif', 'png')
        pred_path = osp.join(pred_dir, pred_name)
        pred_pil = Image.open(pred_path)
        img_list.append(pred_pil)

        save_path = osp.join(save_dir, pred_name)
        vis.splice_imgs(img_list, save_path)
        print('saved in {}'.format(save_path))

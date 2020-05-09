import argparse
import os
import os.path as osp
import cv2
import numpy as np
import tqdm

import utils
import models


def parse_args():
    parser = argparse.ArgumentParser(
        description='HumanSeg inference and visualization')
    parser.add_argument(
        '--test_model',
        dest='test_model',
        help='model path for inference',
        type=str)
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        help='the root directory of dataset',
        type=str)
    parser.add_argument(
        '--file_list', dest='file_list', help='file list for test', type=str)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='the directory for saveing the inferenc results',
        type=str,
        default='./result')
    return parser.parse_args()


def mkdir(path):
    sub_dir = osp.dirname(path)
    if not osp.exists(sub_dir):
        os.makedirs(sub_dir)


def process(args):
    model = models.load_model(args.test_model)
    added_saveed_path = osp.join(args.save_dir, 'added')
    mat_saved_path = osp.join(args.save_dir, 'mat')
    scoremap_saved_path = osp.join(args.save_dir, 'scoremap')

    with open(args.file_list, 'r') as f:
        files = f.readlines()

    for file in tqdm.tqdm(files):
        file = file.strip()
        im_file = osp.join(args.data_dir, file)
        im = cv2.imread(im_file)
        result = model.predict(im)

        # save added image
        added_image = utils.visualize(im_file, result, weight=0.6)
        added_image_file = osp.join(added_saveed_path, file)
        mkdir(added_image_file)
        cv2.imwrite(added_image_file, added_image)

        # save score map
        score_map = result['score_map'][:, :, 1]
        score_map = (score_map * 255).astype(np.uint8)
        score_map_file = osp.join(scoremap_saved_path, file)
        mkdir(score_map_file)
        cv2.imwrite(score_map_file, score_map)

        # save mat image
        score_map = np.expand_dims(score_map, axis=-1)
        mat_image = np.concatenate([im, score_map], axis=2)
        mat_file = osp.join(mat_saved_path, file)
        ext = osp.splitext(mat_file)[-1]
        mat_file = mat_file.replace(ext, '.png')
        mkdir(mat_file)
        cv2.imwrite(mat_file, mat_image)


if __name__ == '__main__':
    args = parse_args()
    process(args)

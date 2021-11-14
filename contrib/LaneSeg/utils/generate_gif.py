import numpy as np
import cv2
import os
import argparse
import imageio
import os.path as ops


def images_to_gif(test_images_root, git_root):
    img_paths = []
    for img_name in os.listdir(test_images_root):
        img_paths.append(ops.join(test_images_root, img_name))
    img_paths.sort()
    gif_frames = []
    for i, img_name in enumerate(img_paths):
        gt_img_org = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
        frame = gt_img_org[..., ::-1]
        frame = frame.astype(np.uint8)
        gif_frames.append(frame)
    imageio.mimsave(git_root, gif_frames, fps=5)


def process_gif(args):
    if not os.path.exists(args.gif_dir):
        os.makedirs(args.gif_dir)
    for dir_name in os.listdir(args.images_root):
        if dir_name[0] == '.':
            continue
        print('Pdrocess the images {} \n'.format(dir_name))
        test_images_root = ops.join(args.images_root, dir_name)
        git_root = ops.join(args.gif_dir, dir_name) + '.gif'
        images_to_gif(test_images_root, git_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--images_root', type=str, default=None, help='The origin path dirs of images')
    parser.add_argument(
        '--gif_dir', type=str, default=None, help='The out gif path')
    args = parser.parse_args()

    process_gif(args)

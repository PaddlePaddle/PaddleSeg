import os
import shutil
from multiprocessing import Pool
from functools import partial

import cv2
import numpy as np
from tqdm import tqdm
"""
Get background from MSCOCO_17 and PascalVOC12 and exclude The images with person.
"""


def get_bg_from_pascal_voc(
        data_path='/mnt/chenguowei01/datasets/VOCdevkit/VOC2012',
        save_path='bg/pascal_val12'):
    """
    extract background
    """
    person_train_txt = os.path.join(data_path,
                                    "ImageSets/Main/person_train.txt")
    train_save_path = os.path.join(save_path, 'train')
    person_val_txt = os.path.join(data_path, "ImageSets/Main/person_val.txt")
    val_save_path = os.path.join(save_path, 'val')
    if not os.path.exists(train_save_path):
        os.makedirs(train_save_path)
    if not os.path.exists(val_save_path):
        os.makedirs(val_save_path)

    # training dataset
    f = open(person_train_txt, 'r')
    train_images = f.read().splitlines()
    f.close()
    print('there are {} images in training dataset.'.format(len(train_images)))
    num = 0
    for line in train_images:
        image_name, id = line.split()
        if id == '-1':
            num += 1
            ori_img = os.path.join(data_path, 'JPEGImages', image_name + '.jpg')
            shutil.copy(ori_img, train_save_path)
    print('there are {} images without person in the training dataset'.format(
        num))

    # val dataset
    f = open(person_val_txt, 'r')
    val_images = f.read().splitlines()
    f.close()
    print('there are {} images in val dataset.'.format(len(val_images)))
    num = 0
    for line in val_images:
        image_name, id = line.split()
        if id == '-1':
            num += 1
            ori_img = os.path.join(data_path, 'JPEGImages', image_name + '.jpg')
            shutil.copy(ori_img, val_save_path)
    print('there are {} images without person in the val dataset'.format(num))


def cp(line, data_path, save_path):
    image_name, anno_name = line.split('|')
    anno = cv2.imread(os.path.join(data_path, anno_name), cv2.IMREAD_UNCHANGED)
    classes = np.unique(anno)
    if 0 not in classes:
        shutil.copy(os.path.join(data_path, image_name), save_path)


def get_bg_from_coco_17(data_path='/mnt/chenguowei01/datasets/coco_17',
                        save_path='bg/coco_17'):
    train_txt = os.path.join(data_path, 'train2017.txt')
    train_save_path = os.path.join(save_path, 'train')
    val_txt = os.path.join(data_path, 'val2017.txt')
    val_save_path = os.path.join(save_path, 'val')
    if not os.path.exists(train_save_path):
        os.makedirs(train_save_path)
    if not os.path.exists(val_save_path):
        os.makedirs(val_save_path)

    # training dataset
    partial_train_cp = partial(
        cp, data_path=data_path, save_path=train_save_path)
    with open(train_txt, 'r') as f:
        train_list = f.read().splitlines()
    max_ = len(train_list)
    with Pool(40) as pool:
        with tqdm(total=max_) as pbar:
            for i, _ in tqdm(
                    enumerate(
                        pool.imap_unordered(partial_train_cp, train_list))):
                pbar.update()

    # val dataset
    partial_val_cp = partial(cp, data_path=data_path, save_path=val_save_path)
    with open(val_txt, 'r') as f:
        val_list = f.read().splitlines()
    max_ = len(val_list)
    with Pool(40) as pool:
        with tqdm(total=max_) as pbar:
            for i, _ in tqdm(
                    enumerate(pool.imap_unordered(partial_val_cp, val_list))):
                pbar.update()


if __name__ == "__main__":
    # get_bg_from_pascal_voc(save_path="/mnt/chenguowei01/datasets/matting/gather/bg/pascal_voc12")
    get_bg_from_coco_17(
        save_path="/mnt/chenguowei01/datasets/matting/gather/bg/coco_17")

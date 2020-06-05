# coding: utf8
# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import os.path as osp
import sys
import argparse
from PIL import Image
from tqdm import tqdm
import imghdr
import logging
import pickle
import gdal


def parse_args():
    parser = argparse.ArgumentParser(
        description='Data analyse and data check before training.')
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        help='Dataset directory',
        default=None,
        type=str)
    parser.add_argument(
        '--num_classes',
        dest='num_classes',
        help='Number of classes',
        default=None,
        type=int)
    parser.add_argument(
        '--separator',
        dest='separator',
        help='file list separator',
        default=" ",
        type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def init_global_variable():
    """
    初始化全局变量
    """
    global png_format_right_num  # 格式正确的标注图数量
    global png_format_wrong_num  # 格式错误的标注图数量
    global total_grt_classes  # 标注类别
    global total_num_of_each_class  # 标注每个类别总的像素数
    global shape_unequal_image  # 图片和标注shape不一致列表
    global png_format_wrong_image  # 标注格式错误列表
    global max_width  # 图片最长宽
    global max_height  # 图片最长高
    global min_width  # 图片最短宽
    global min_height  # 图片最短高
    global min_aspectratio  # 图片最小宽高比
    global max_aspectratio  # 图片最大宽高比
    global img_dim  # 图片的通道数
    global list_wrong  # 文件名格式错误列表
    global imread_failed  # 图片读取失败列表, 二元列表
    global label_wrong  # 标注图片出错列表
    global label_not_single_channel  # 标注图非灰度图列表

    png_format_right_num = 0
    png_format_wrong_num = 0
    total_grt_classes = []
    total_num_of_each_class = []
    shape_unequal_image = []
    png_format_wrong_image = []
    max_width = 0
    max_height = 0
    min_width = sys.float_info.max
    min_height = sys.float_info.max
    min_aspectratio = sys.float_info.max
    max_aspectratio = 0
    img_dim = []
    list_wrong = []
    imread_failed = []
    label_wrong = []
    label_not_single_channel = []


def read_img(img_path):
    img_format = imghdr.what(img_path)
    name, ext = osp.splitext(img_path)
    if img_format == 'tiff' or ext == '.img':
        dataset = gdal.Open(img_path)
        if dataset == None:
            raise Exception('Can not open', img_path)
        im_data = dataset.ReadAsArray()
        return im_data.transpose((1, 2, 0))
    elif ext == '.npy':
        return np.load(img_path)
    else:
        raise Exception('Not support {} image format!'.format(ext))


def img_pixel_statistics(img):
    global IMG_VALUE_NUM, MEANS, STDS, TOTAL_IMG_NUM, CLIP_MIN_VALUE, CLIP_MAX_VALUE, IMG_MIN_VALUE, IMG_MAX_VALUE

    TOTAL_IMG_NUM += 1
    channel = img.shape[2]
    if MEANS == []:
        MEANS = [0] * channel
    if STDS == []:
        STDS = [0] * channel
    if IMG_MIN_VALUE == []:
        IMG_MIN_VALUE = [sys.float_info.max] * channel
    if IMG_MAX_VALUE == []:
        IMG_MAX_VALUE = [0] * channel
    if IMG_VALUE_NUM == []:
        [IMG_VALUE_NUM.append([]) for i in range(channel)]
    for k in range(channel):
        img_k = img[:, :, k]

        # count mean, std
        img_mean = np.mean(img_k)
        img_std = np.std(img_k)
        MEANS[k] += img_mean
        STDS[k] += img_std

        # count min, max
        min_value = np.min(img_k)
        max_value = np.max(img_k)
        if IMG_MAX_VALUE[k] < max_value:
            IMG_MAX_VALUE[k] = max_value
        if IMG_MIN_VALUE[k] > min_value:
            IMG_MIN_VALUE[k] = min_value

        # count the distribution of image value, value number
        unique, counts = np.unique(img_k, return_counts=True)
        add_num = []
        max_unique = np.max(unique)
        add_len = max_unique - len(IMG_VALUE_NUM[k]) + 1
        if add_len > 0:
            IMG_VALUE_NUM[k] += ([0] * add_len)
        for i in range(len(unique)):
            value = unique[i]
            IMG_VALUE_NUM[k][value] += counts[i]

        IMG_VALUE_NUM[k] += add_num


def dataset_pixel_statistics():
    logger.info(
        "\n-----------------------------\n4. Dataset pixel statistics...")
    global MEANS, STDS

    # count the distribution of image value, value number
    if IMG_VALUE_NUM == []:
        return
    logger.info("\nImage pixel statistics:")
    total_ratio = []
    [total_ratio.append([]) for i in range(len(IMG_VALUE_NUM))]
    for k in range(len(IMG_VALUE_NUM)):
        total_num = sum(IMG_VALUE_NUM[k])
        total_ratio[k] = [i / total_num for i in IMG_VALUE_NUM[k]]
        total_ratio[k] = np.around(total_ratio[k], decimals=4)
    with open(os.path.join(DATA_DIR, 'img_pixel_statistics.pkl'), 'wb') as f:
        pickle.dump([total_ratio, IMG_VALUE_NUM], f)

    # print min value, max value
    logger.info("value range: \nIMG_MIN_VALUE = {} \nIMG_MAX_VALUE = {}".format(
        IMG_MIN_VALUE, IMG_MAX_VALUE))

    # count mean, std
    MEANS = [i / TOTAL_IMG_NUM for i in MEANS]
    STDS = [i / TOTAL_IMG_NUM for i in STDS]
    logger.info("\nCount the channel-by-channel mean and std of the image:\n"
                "mean = {}\nstd = {}".format(MEANS, STDS))


def error_print(str):
    return "".join(["\nNOT PASS ", str])


def correct_print(str):
    return "".join(["\nPASS ", str])


def pil_imread(file_path):
    """read pseudo-color label"""
    im = Image.open(file_path)
    return np.asarray(im)


def get_img_shape_range(img):
    """获取图片最大和最小宽高"""
    global max_width, max_height, min_width, min_height
    img_shape = img.shape
    height, width = img_shape[0], img_shape[1]
    max_height = max(height, max_height)
    max_width = max(width, max_width)
    min_height = min(height, min_height)
    min_width = min(width, min_width)


def get_image_min_max_aspectratio(img):
    """计算图片最大宽高比"""
    global min_aspectratio, max_aspectratio
    img_shape = img.shape
    height, width = img_shape[0], img_shape[1]
    min_aspectratio = min(width / height, min_aspectratio)
    max_aspectratio = max(width / height, max_aspectratio)
    return min_aspectratio, max_aspectratio


def get_image_dim(img):
    """获取图像的通道数"""
    img_shape = img.shape
    if img_shape[-1] not in img_dim:
        img_dim.append(img_shape[-1])


def is_label_single_channel(grt):
    """判断标签是否为灰度图"""
    grt_shape = grt.shape
    if len(grt_shape) == 2:
        return True
    else:
        return False


def image_label_shape_check(img, grt):
    """
    验证图像和标注的大小是否匹配
    """

    flag = True
    img_height = img.shape[0]
    img_width = img.shape[1]
    grt_height = grt.shape[0]
    grt_width = grt.shape[1]

    if img_height != grt_height or img_width != grt_width:
        flag = False
    return flag


def ground_truth_check(grt, grt_path):
    """
    验证标注图像的格式
    统计标注图类别和像素数
    params:
        grt: 标注图
        grt_path: 标注图路径
    return:
        png_format: 返回是否是png格式图片
        unique: 返回标注类别
        counts: 返回标注的像素数
    """
    if imghdr.what(grt_path) == "png":
        png_format = True
    else:
        png_format = False

    unique, counts = np.unique(grt, return_counts=True)

    return png_format, unique, counts


def sum_gt_check(png_format, grt_classes, num_of_each_class):
    """
    统计所有标注图上的格式、类别和每个类别的像素数
    params:
        png_format: 是否是png格式图片
        grt_classes: 标注类别
        num_of_each_class: 各个类别的像素数目
    """
    is_label_correct = True
    global png_format_right_num, png_format_wrong_num, total_grt_classes, total_num_of_each_class

    if png_format:
        png_format_right_num += 1
    else:
        png_format_wrong_num += 1

    if IGNORE_INDEX in grt_classes:
        grt_classes2 = np.delete(grt_classes,
                                 np.where(grt_classes == IGNORE_INDEX))
    else:
        grt_classes2 = grt_classes
    if min(grt_classes2) < 0 or max(grt_classes2) > NUM_CLASSES - 1:
        is_label_correct = False
    add_class = []
    add_num = []
    for i in range(len(grt_classes)):
        gi = grt_classes[i]
        if gi in total_grt_classes:
            j = total_grt_classes.index(gi)
            total_num_of_each_class[j] += num_of_each_class[i]
        else:
            add_class.append(gi)
            add_num.append(num_of_each_class[i])
    total_num_of_each_class += add_num
    total_grt_classes += add_class
    return is_label_correct


def label_check_statistics():
    """
    对标注图像进行校验，输出校验结果
    """
    if png_format_wrong_num == 0:
        if png_format_right_num:
            logger.info(correct_print("label format check"))
        else:
            logger.info(error_print("label format check"))
            logger.info("No label image to check")
            return
    else:
        logger.info(error_print("label format check"))
    logger.info(
        "total {} label images are png format, {} label images are not png "
        "format".format(png_format_right_num, png_format_wrong_num))
    if len(png_format_wrong_image) > 0:
        for i in png_format_wrong_image:
            logger.debug(i)

    total_ratio = total_num_of_each_class / sum(total_num_of_each_class)
    total_ratio = np.around(total_ratio, decimals=4)
    total_nc = sorted(
        zip(total_grt_classes, total_ratio, total_num_of_each_class))
    if len(label_wrong) == 0 and not total_nc[0][0]:
        logger.info(correct_print("label class check!"))
    else:
        logger.info(error_print("label class check!"))
        if total_nc[0][0]:
            logger.info("Warning: label classes should start from 0")
        if len(label_wrong) > 0:
            logger.info(
                "fatal error: label class is out of range [0, {}]".format(
                    NUM_CLASSES - 1))
            for i in label_wrong:
                logger.debug(i)

    logger.info(
        "\nLabel pixel statistics:\n"
        "(label class, percentage, total pixel number) = {} ".format(total_nc))


def shape_check():
    """输出shape校验结果"""
    if len(shape_unequal_image) == 0:
        logger.info(correct_print("shape check"))
        logger.info("All images are the same shape as the labels")
    else:
        logger.info(error_print("shape check"))
        logger.info(
            "Some images are not the same shape as the labels as follow: ")
        for i in shape_unequal_image:
            logger.debug(i)


def file_list_check(list_name):
    """检查分割符是否复合要求"""
    if len(list_wrong) == 0:
        logger.info(
            correct_print(
                list_name.split(os.sep)[-1] + " DATASET.SEPARATOR check"))
    else:
        logger.info(
            error_print(
                list_name.split(os.sep)[-1] + " DATASET.SEPARATOR check"))
        logger.info(
            "The following list is not separated by {}".format(SEPARATOR))
        for i in list_wrong:
            logger.debug(i)


def imread_check():
    if len(imread_failed) == 0:
        logger.info(correct_print("dataset reading check"))
        logger.info("All images can be read successfully")
    else:
        logger.info(error_print("dataset reading check"))
        logger.info("Failed to read {} images".format(len(imread_failed)))
        for i in imread_failed:
            logger.debug(i)


def single_channel_label_check():
    if len(label_not_single_channel) == 0:
        logger.info(correct_print("label single_channel check"))
        logger.info("All label images are single_channel")
    else:
        logger.info(error_print("label single_channel check"))
        logger.info(
            "{} label images are not single_channel\nLabel pixel statistics may be insignificant"
            .format(len(label_not_single_channel)))
        for i in label_not_single_channel:
            logger.debug(i)


def img_shape_range_statistics():
    logger.info("\nImage size statistics:")
    logger.info(
        "max width = {}  min width = {}  max height = {}  min height = {}".
        format(max_width, min_width, max_height, min_height))


def img_dim_statistics():
    logger.info("\nImage channels statistics\nImage channels = {}".format(
        np.unique(img_dim)))


def check_train_dataset():
    list_file = TRAIN_FILE_LIST
    logger.info("-----------------------------\n1. Check train dataset...")
    with open(list_file, 'r') as fid:
        lines = fid.readlines()
        if not lines:
            print("File list is empty!")
            return
        for line in tqdm(lines):
            line = line.strip()
            parts = line.split(SEPARATOR)
            if len(parts) != 2:
                list_wrong.append(line)
                continue
            img_name, grt_name = parts[0], parts[1]
            img_path = os.path.join(DATA_DIR, img_name)
            grt_path = os.path.join(DATA_DIR, grt_name)
            try:
                img = read_img(img_path)
                grt = pil_imread(grt_path)
            except Exception as e:
                imread_failed.append((line, str(e)))
                continue

            img_pixel_statistics(img)

            is_single_channel = is_label_single_channel(grt)
            if not is_single_channel:
                label_not_single_channel.append(line)
                continue
            get_img_shape_range(img)
            get_image_dim(img)
            is_equal_img_grt_shape = image_label_shape_check(img, grt)
            if not is_equal_img_grt_shape:
                shape_unequal_image.append(line)

            png_format, grt_classes, num_of_each_class = ground_truth_check(
                grt, grt_path)
            if not png_format:
                png_format_wrong_image.append(line)
            is_label_correct = sum_gt_check(png_format, grt_classes,
                                            num_of_each_class)
            if not is_label_correct:
                label_wrong.append(line)

        file_list_check(list_file)
        imread_check()
        single_channel_label_check()
        shape_check()
        label_check_statistics()

        img_dim_statistics()
        img_shape_range_statistics()


def check_val_dataset():
    list_file = VAL_FILE_LIST
    logger.info("\n-----------------------------\n2. Check val dataset...")
    with open(list_file) as fid:
        lines = fid.readlines()
        if not lines:
            print("File list is empty!")
            return
        for line in tqdm(lines):
            line = line.strip()
            parts = line.split(SEPARATOR)
            if len(parts) != 2:
                list_wrong.append(line)
                continue
            img_name, grt_name = parts[0], parts[1]
            img_path = os.path.join(DATA_DIR, img_name)
            grt_path = os.path.join(DATA_DIR, grt_name)
            try:
                img = read_img(img_path)
                grt = pil_imread(grt_path)
            except Exception as e:
                imread_failed.append((line, str(e)))
                continue

            img_pixel_statistics(img)

            is_single_channel = is_label_single_channel(grt)
            if not is_single_channel:
                label_not_single_channel.append(line)
                continue
            get_img_shape_range(img)
            get_image_min_max_aspectratio(img)
            get_image_dim(img)
            is_equal_img_grt_shape = image_label_shape_check(img, grt)
            if not is_equal_img_grt_shape:
                shape_unequal_image.append(line)
            png_format, grt_classes, num_of_each_class = ground_truth_check(
                grt, grt_path)
            if not png_format:
                png_format_wrong_image.append(line)
            is_label_correct = sum_gt_check(png_format, grt_classes,
                                            num_of_each_class)
            if not is_label_correct:
                label_wrong.append(line)

        file_list_check(list_file)
        imread_check()
        single_channel_label_check()
        shape_check()
        label_check_statistics()

        img_dim_statistics()
        img_shape_range_statistics()


def check_test_dataset():
    list_file = TEST_FILE_LIST
    has_label = False
    with open(list_file) as fid:
        logger.info("\n-----------------------------\n3. Check test dataset...")
        lines = fid.readlines()
        if not lines:
            print("File list is empty!")
            return
        for line in tqdm(lines):
            line = line.strip()
            parts = line.split(SEPARATOR)
            if len(parts) == 1:
                img_name = parts
                img_path = os.path.join(DATA_DIR, img_name[0])
                try:
                    img = read_img(img_path)
                except Exception as e:
                    imread_failed.append((line, str(e)))
                    continue
            elif len(parts) == 2:
                has_label = True
                img_name, grt_name = parts[0], parts[1]
                img_path = os.path.join(DATA_DIR, img_name)
                grt_path = os.path.join(DATA_DIR, grt_name)
                try:
                    img = read_img(img_path)
                    grt = pil_imread(grt_path)
                except Exception as e:
                    imread_failed.append((line, str(e)))
                    continue

                img_pixel_statistics(img)

                is_single_channel = is_label_single_channel(grt)
                if not is_single_channel:
                    label_not_single_channel.append(line)
                    continue
                is_equal_img_grt_shape = image_label_shape_check(img, grt)
                if not is_equal_img_grt_shape:
                    shape_unequal_image.append(line)
                png_format, grt_classes, num_of_each_class = ground_truth_check(
                    grt, grt_path)
                if not png_format:
                    png_format_wrong_image.append(line)
                is_label_correct = sum_gt_check(png_format, grt_classes,
                                                num_of_each_class)
                if not is_label_correct:
                    label_wrong.append(line)
            else:
                list_wrong.append(lines)
                continue
            get_img_shape_range(img)
            get_image_min_max_aspectratio(img)
            get_image_dim(img)

        file_list_check(list_file)
        imread_check()
        img_dim_statistics()
        img_shape_range_statistics()

        if has_label:
            single_channel_label_check()
            shape_check()
            label_check_statistics()


def main():
    init_global_variable()
    check_train_dataset()

    init_global_variable()
    check_val_dataset()

    init_global_variable()
    check_test_dataset()

    dataset_pixel_statistics()

    print("\nDetailed error information can be viewed in {}.".format(
        os.path.join(DATA_DIR, 'data_analyse_and_check.log')))


if __name__ == "__main__":
    args = parse_args()
    DATA_DIR = args.data_dir
    TRAIN_FILE_LIST = osp.join(DATA_DIR, 'train.txt')
    VAL_FILE_LIST = osp.join(DATA_DIR, 'val.txt')
    TEST_FILE_LIST = osp.join(DATA_DIR, 'test.txt')
    IGNORE_INDEX = 255
    NUM_CLASSES = args.num_classes
    SEPARATOR = args.separator
    IMG_MIN_VALUE = []
    IMG_MAX_VALUE = []
    MEANS = []
    STDS = []
    TOTAL_IMG_NUM = 0
    IMG_VALUE_NUM = []

    logger = logging.getLogger()
    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(message)s"
    formatter = logging.Formatter(BASIC_FORMAT)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel('INFO')
    th = logging.FileHandler(
        os.path.join(DATA_DIR, 'data_analyse_and_check.log'), 'w')
    th.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(th)

    main()

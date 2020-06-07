# coding: utf8
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


def img_pixel_statistics(img, img_value_num, img_min_value, img_max_value):
    channel = img.shape[2]
    means = np.zeros(channel)
    stds = np.zeros(channel)
    for k in range(channel):
        img_k = img[:, :, k]

        # count mean, std
        means[k] = np.mean(img_k)
        stds[k] = np.std(img_k)

        # count min, max
        min_value = np.min(img_k)
        max_value = np.max(img_k)
        if img_max_value[k] < max_value:
            img_max_value[k] = max_value
        if img_min_value[k] > min_value:
            img_min_value[k] = min_value

        # count the distribution of image value, value number
        unique, counts = np.unique(img_k, return_counts=True)
        add_num = []
        max_unique = np.max(unique)
        add_len = max_unique - len(img_value_num[k]) + 1
        if add_len > 0:
            img_value_num[k] += ([0] * add_len)
        for i in range(len(unique)):
            value = unique[i]
            img_value_num[k][value] += counts[i]

        img_value_num[k] += add_num
    return means, stds, img_min_value, img_max_value, img_value_num


def dataset_pixel_statistics(data_dir, total_means, total_stds, img_value_num,
                             img_min_value, img_max_value, total_img_num,
                             logger):
    logger.info("\n-----------------------------\nDataset pixel statistics...")

    # count the distribution of image value, value number
    if not img_value_num:
        return
    logger.info("\nImage pixel statistics:")
    total_ratio = []
    [total_ratio.append([]) for i in range(len(img_value_num))]
    for k in range(len(img_value_num)):
        total_num = sum(img_value_num[k])
        total_ratio[k] = [i / total_num for i in img_value_num[k]]
        total_ratio[k] = np.around(total_ratio[k], decimals=4)
    with open(os.path.join(data_dir, 'img_pixel_statistics.pkl'), 'wb') as f:
        pickle.dump([total_ratio, img_value_num], f)

    # print min value, max value
    logger.info("value range: \nimg_min_value = {} \nimg_max_value = {}".format(
        img_min_value, img_max_value))

    # count mean, std
    total_means = total_means / total_img_num
    total_stds = total_stds / total_img_num
    print("\nCount the channel-by-channel mean and std of the image:\n"
          "mean = {}\nstd = {}".format(total_means, total_stds))


def error_print(str):
    return "".join(["\nNOT PASS ", str])


def correct_print(str):
    return "".join(["\nPASS ", str])


def pil_imread(file_path):
    """read pseudo-color label"""
    im = Image.open(file_path)
    return np.asarray(im)


def get_img_shape_range(img, max_width, max_height, min_width, min_height):
    """获取图片最大和最小宽高"""
    img_shape = img.shape
    height, width = img_shape[0], img_shape[1]
    max_height = max(height, max_height)
    max_width = max(width, max_width)
    min_height = min(height, min_height)
    min_width = min(width, min_width)
    return max_width, max_height, min_width, min_height


def get_image_dim(img, img_dim):
    """获取图像的通道数"""
    img_shape = img.shape
    if img_shape[-1] not in img_dim:
        img_dim.append(img_shape[-1])
    return img_dim


def is_label_single_channel(label):
    """判断标签是否为灰度图"""
    label_shape = label.shape
    if len(label_shape) == 2:
        return True
    else:
        return False


def image_label_shape_check(img, label):
    """
    验证图像和标注的大小是否匹配
    """

    flag = True
    img_height = img.shape[0]
    img_width = img.shape[1]
    label_height = label.shape[0]
    label_width = label.shape[1]

    if img_height != label_height or img_width != label_width:
        flag = False
    return flag


def ground_truth_check(label, label_path):
    """
    验证标注图像的格式
    统计标注图类别和像素数
    params:
        label: 标注图
        label_path: 标注图路径
    return:
        png_format: 返回是否是png格式图片
        unique: 返回标注类别
        counts: 返回标注的像素数
    """
    if imghdr.what(label_path) == "png":
        png_format = True
    else:
        png_format = False

    unique, counts = np.unique(label, return_counts=True)

    return png_format, unique, counts


def sum_label_check(png_format, label_classes, num_of_each_class, ignore_index,
                    num_classes, png_format_right_num, png_format_wrong_num,
                    total_label_classes, total_num_of_each_class):
    """
    统计所有标注图上的格式、类别和每个类别的像素数
    params:
        png_format: 是否是png格式图片
        label_classes: 标注类别
        num_of_each_class: 各个类别的像素数目
    """
    is_label_correct = True

    if png_format:
        png_format_right_num += 1
    else:
        png_format_wrong_num += 1

    if ignore_index in label_classes:
        label_classes2 = np.delete(label_classes,
                                   np.where(label_classes == ignore_index))
    else:
        label_classes2 = label_classes
    if min(label_classes2) < 0 or max(label_classes2) > num_classes - 1:
        is_label_correct = False
    add_class = []
    add_num = []
    for i in range(len(label_classes)):
        gi = label_classes[i]
        if gi in total_label_classes:
            j = total_label_classes.index(gi)
            total_num_of_each_class[j] += num_of_each_class[i]
        else:
            add_class.append(gi)
            add_num.append(num_of_each_class[i])
    total_num_of_each_class += add_num
    total_label_classes += add_class
    return is_label_correct, png_format_right_num, png_format_wrong_num, total_num_of_each_class, total_label_classes


def label_check_statistics(num_classes, png_format_wrong_image,
                           png_format_right_num, png_format_wrong_num,
                           total_label_classes, total_num_of_each_class,
                           wrong_labels, logger):
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
        zip(total_label_classes, total_ratio, total_num_of_each_class))
    if len(wrong_labels) == 0 and not total_nc[0][0]:
        logger.info(correct_print("label class check!"))
    else:
        logger.info(error_print("label class check!"))
        if total_nc[0][0]:
            logger.info("Warning: label classes should start from 0")
        if len(wrong_labels) > 0:
            logger.info(
                "fatal error: label class is out of range [0, {}]".format(
                    num_classes - 1))
            for i in wrong_labels:
                logger.debug(i)

    logger.info(
        "\nLabel pixel statistics:\n"
        "(label class, percentage, total pixel number) = {} ".format(total_nc))


def shape_check(shape_unequal_image, logger):
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


def separator_check(wrong_lines, file_list, separator, logger):
    """检查分割符是否复合要求"""
    if len(wrong_lines) == 0:
        logger.info(
            correct_print(
                file_list.split(os.sep)[-1] + " DATASET.separator check"))
    else:
        logger.info(
            error_print(
                file_list.split(os.sep)[-1] + " DATASET.separator check"))
        logger.info(
            "The following list is not separated by {}".format(separator))
        for i in wrong_lines:
            logger.debug(i)


def imread_check(imread_failed, logger):
    if len(imread_failed) == 0:
        logger.info(correct_print("dataset reading check"))
        logger.info("All images can be read successfully")
    else:
        logger.info(error_print("dataset reading check"))
        logger.info("Failed to read {} images".format(len(imread_failed)))
        for i in imread_failed:
            logger.debug(i)


def single_channel_label_check(label_not_single_channel, logger):
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


def img_shape_range_statistics(max_width, min_width, max_height, min_height,
                               logger):
    logger.info("\nImage size statistics:")
    logger.info(
        "max width = {}  min width = {}  max height = {}  min height = {}".
        format(max_width, min_width, max_height, min_height))


def img_dim_statistics(img_dim, logger):
    logger.info("\nImage channels statistics\nImage channels = {}".format(
        np.unique(img_dim)))


def data_analyse_and_check(data_dir, num_classes, separator, ignore_index,
                           logger):
    train_file_list = osp.join(data_dir, 'train.txt')
    val_file_list = osp.join(data_dir, 'val.txt')
    test_file_list = osp.join(data_dir, 'test.txt')
    total_img_num = 0
    has_label = False
    for file_list in [train_file_list, val_file_list, test_file_list]:
        # initialization
        imread_failed = []
        max_width = 0
        max_height = 0
        min_width = sys.float_info.max
        min_height = sys.float_info.max
        label_not_single_channel = []
        shape_unequal_image = []
        png_format_wrong_image = []
        wrong_labels = []
        wrong_lines = []
        png_format_right_num = 0
        png_format_wrong_num = 0
        total_label_classes = []
        total_num_of_each_class = []
        img_dim = []

        with open(file_list, 'r') as fid:
            logger.info("\n-----------------------------\nCheck {}...".format(
                file_list))
            lines = fid.readlines()
            if not lines:
                logger.info("File list is empty!")
                continue
            for line in tqdm(lines):
                line = line.strip()
                parts = line.split(separator)
                if len(parts) == 1:
                    if file_list == train_file_list or file_list == val_file_list:
                        logger.info("Train or val list must have labels!")
                        break
                    img_name = parts
                    img_path = os.path.join(data_dir, img_name[0])
                    try:
                        img = read_img(img_path)
                    except Exception as e:
                        imread_failed.append((line, str(e)))
                        continue
                elif len(parts) == 2:
                    has_label = True
                    img_name, label_name = parts[0], parts[1]
                    img_path = os.path.join(data_dir, img_name)
                    label_path = os.path.join(data_dir, label_name)
                    try:
                        img = read_img(img_path)
                        label = pil_imread(label_path)
                    except Exception as e:
                        imread_failed.append((line, str(e)))
                        continue

                    is_single_channel = is_label_single_channel(label)
                    if not is_single_channel:
                        label_not_single_channel.append(line)
                        continue
                    is_equal_img_label_shape = image_label_shape_check(
                        img, label)
                    if not is_equal_img_label_shape:
                        shape_unequal_image.append(line)
                    png_format, label_classes, num_of_each_class = ground_truth_check(
                        label, label_path)
                    if not png_format:
                        png_format_wrong_image.append(line)
                    is_label_correct, png_format_right_num, png_format_wrong_num, total_num_of_each_class, total_label_classes = sum_label_check(
                        png_format, label_classes, num_of_each_class,
                        ignore_index, num_classes, png_format_right_num,
                        png_format_wrong_num, total_label_classes,
                        total_num_of_each_class)
                    if not is_label_correct:
                        wrong_labels.append(line)
                else:
                    wrong_lines.append(lines)
                    continue

                if total_img_num == 0:
                    channel = img.shape[2]
                    total_means = np.zeros(channel)
                    total_stds = np.zeros(channel)
                    img_min_value = [sys.float_info.max] * channel
                    img_max_value = [0] * channel
                    img_value_num = []
                    [img_value_num.append([]) for i in range(channel)]
                means, stds, img_min_value, img_max_value, img_value_num = img_pixel_statistics(
                    img, img_value_num, img_min_value, img_max_value)
                total_means += means
                total_stds += stds
                max_width, max_height, min_width, min_height = get_img_shape_range(
                    img, max_width, max_height, min_width, min_height)
                img_dim = get_image_dim(img, img_dim)
                total_img_num += 1

            separator_check(wrong_lines, file_list, separator, logger)
            imread_check(imread_failed, logger)
            img_dim_statistics(img_dim, logger)
            img_shape_range_statistics(max_width, min_width, max_height,
                                       min_height, logger)

            if has_label:
                single_channel_label_check(label_not_single_channel, logger)
                shape_check(shape_unequal_image, logger)
                label_check_statistics(
                    num_classes, png_format_wrong_image, png_format_right_num,
                    png_format_wrong_num, total_label_classes,
                    total_num_of_each_class, wrong_labels, logger)

    dataset_pixel_statistics(data_dir, total_means, total_stds, img_value_num,
                             img_min_value, img_max_value, total_img_num,
                             logger)


def main():
    args = parse_args()
    data_dir = args.data_dir
    ignore_index = 255
    num_classes = args.num_classes
    separator = args.separator

    logger = logging.getLogger()
    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(message)s"
    formatter = logging.Formatter(BASIC_FORMAT)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel('INFO')
    th = logging.FileHandler(
        os.path.join(data_dir, 'data_analyse_and_check.log'), 'w')
    th.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(th)

    data_analyse_and_check(data_dir, num_classes, separator, ignore_index,
                           logger)

    print("\nDetailed error information can be viewed in {}.".format(
        os.path.join(data_dir, 'data_analyse_and_check.log')))


if __name__ == "__main__":
    main()

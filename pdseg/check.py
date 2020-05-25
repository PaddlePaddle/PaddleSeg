# coding: utf8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
import sys
import pprint
import argparse
import cv2
from tqdm import tqdm
import imghdr
import logging

from utils.config import cfg
from reader import pil_imread


def init_global_variable():
    """
    初始化全局变量
    """
    global png_format_right_num  # 格式正确的标注图数量
    global png_format_wrong_num  # 格式错误的标注图数量
    global total_grt_classes  # 总的标注类别
    global total_num_of_each_class  # 每个类别总的像素数
    global shape_unequal_image  # 图片和标注shape不一致列表
    global png_format_wrong_image  # 标注格式错误列表
    global max_width  # 图片最长宽
    global max_height  # 图片最长高
    global min_aspectratio  # 图片最小宽高比
    global max_aspectratio  # 图片最大宽高比
    global img_dim  # 图片的通道数
    global list_wrong  # 文件名格式错误列表
    global imread_failed  # 图片读取失败列表, 二元列表
    global label_wrong  # 标注图片出错列表
    global label_gray_wrong  # 标注图非灰度图列表

    png_format_right_num = 0
    png_format_wrong_num = 0
    total_grt_classes = []
    total_num_of_each_class = []
    shape_unequal_image = []
    png_format_wrong_image = []
    max_width = 0
    max_height = 0
    min_aspectratio = sys.float_info.max
    max_aspectratio = 0
    img_dim = []
    list_wrong = []
    imread_failed = []
    label_wrong = []
    label_gray_wrong = []


def parse_args():
    parser = argparse.ArgumentParser(description='PaddleSeg check')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default=None,
        type=str)
    return parser.parse_args()


def error_print(str):
    return "".join(["\nNOT PASS ", str])


def correct_print(str):
    return "".join(["\nPASS ", str])


def cv2_imread(file_path, flag=cv2.IMREAD_COLOR):
    """
    解决 cv2.imread 在window平台打开中文路径的问题.
    """
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flag)


def get_image_max_height_width(img):
    """获取图片最大宽和高"""
    global max_width, max_height
    img_shape = img.shape
    height, width = img_shape[0], img_shape[1]
    max_height = max(height, max_height)
    max_width = max(width, max_width)


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


def is_label_gray(grt):
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

    if cfg.DATASET.IGNORE_INDEX in grt_classes:
        grt_classes2 = np.delete(
            grt_classes, np.where(grt_classes == cfg.DATASET.IGNORE_INDEX))
    else:
        grt_classes2 = grt_classes
    if min(grt_classes2) < 0 or max(grt_classes2) > cfg.DATASET.NUM_CLASSES - 1:
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


def gt_check():
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
        zip(total_grt_classes, total_num_of_each_class, total_ratio))
    logger.info(
        "\nDoing label pixel statistics:\n"
        "(label class, total pixel number, percentage) = {} ".format(total_nc))

    if len(label_wrong) == 0 and not total_nc[0][0]:
        logger.info(correct_print("label class check!"))
    else:
        logger.info(error_print("label class check!"))
        if total_nc[0][0]:
            logger.info("Warning: label classes should start from 0")
        if len(label_wrong) > 0:
            logger.info(
                "fatal error: label class is out of range [0, {}]".format(
                    cfg.DATASET.NUM_CLASSES - 1))
            for i in label_wrong:
                logger.debug(i)


def eval_crop_size_check(max_height, max_width, min_aspectratio,
                         max_aspectratio):
    """
    判断eval_crop_siz与验证集及测试集的max_height, max_width的关系
    param
        max_height: 数据集的最大高
        max_width: 数据集的最大宽
    """

    if cfg.AUG.AUG_METHOD == "stepscaling":
        if max_width <= cfg.EVAL_CROP_SIZE[
                0] and max_height <= cfg.EVAL_CROP_SIZE[1]:
            logger.info(correct_print("EVAL_CROP_SIZE check"))
            logger.info(
                "satisfy current EVAL_CROP_SIZE: ({},{}) >= max width and max height of images: ({},{})"
                .format(cfg.EVAL_CROP_SIZE[0], cfg.EVAL_CROP_SIZE[1], max_width,
                        max_height))
        else:
            logger.info(error_print("EVAL_CROP_SIZE check"))
            if max_width > cfg.EVAL_CROP_SIZE[0]:
                logger.info(
                    "EVAL_CROP_SIZE[0]: {} should >= max width of images {}!".
                    format(cfg.EVAL_CROP_SIZE[0], max_width))
            if max_height > cfg.EVAL_CROP_SIZE[1]:
                logger.info(
                    "EVAL_CROP_SIZE[1]: {} should >= max height of images {}!".
                    format(cfg.EVAL_CROP_SIZE[1], max_height))

    elif cfg.AUG.AUG_METHOD == "rangescaling":
        if min_aspectratio <= 1 and max_aspectratio >= 1:
            if cfg.EVAL_CROP_SIZE[0] >= cfg.AUG.INF_RESIZE_VALUE \
                    and cfg.EVAL_CROP_SIZE[1] >= cfg.AUG.INF_RESIZE_VALUE:
                logger.info(correct_print("EVAL_CROP_SIZE check"))
                logger.info(
                    "satisfy current EVAL_CROP_SIZE: ({},{}) >= ({},{}) ".
                    format(cfg.EVAL_CROP_SIZE[0], cfg.EVAL_CROP_SIZE[1],
                           cfg.AUG.INF_RESIZE_VALUE, cfg.AUG.INF_RESIZE_VALUE))
            else:
                logger.info(error_print("EVAL_CROP_SIZE check"))
                logger.info(
                    "EVAL_CROP_SIZE must >= img size({},{}), current EVAL_CROP_SIZE is ({},{})"
                    .format(cfg.AUG.INF_RESIZE_VALUE, cfg.AUG.INF_RESIZE_VALUE,
                            cfg.EVAL_CROP_SIZE[0], cfg.EVAL_CROP_SIZE[1]))
        elif min_aspectratio > 1:
            max_height_rangscaling = cfg.AUG.INF_RESIZE_VALUE / min_aspectratio
            max_height_rangscaling = round(max_height_rangscaling)
            if cfg.EVAL_CROP_SIZE[
                    0] >= cfg.AUG.INF_RESIZE_VALUE and cfg.EVAL_CROP_SIZE[
                        1] >= max_height_rangscaling:
                logger.info(correct_print("EVAL_CROP_SIZE check"))
                logger.info(
                    "satisfy current EVAL_CROP_SIZE: ({},{}) >= ({},{}) ".
                    format(cfg.EVAL_CROP_SIZE[0], cfg.EVAL_CROP_SIZE[1],
                           cfg.AUG.INF_RESIZE_VALUE, max_height_rangscaling))
            else:
                logger.info(error_print("EVAL_CROP_SIZE check"))
                logger.info(
                    "EVAL_CROP_SIZE must >= img size({},{}), current EVAL_CROP_SIZE is ({},{})"
                    .format(cfg.AUG.INF_RESIZE_VALUE, max_height_rangscaling,
                            cfg.EVAL_CROP_SIZE[0], cfg.EVAL_CROP_SIZE[1]))
        elif max_aspectratio < 1:
            max_width_rangscaling = cfg.AUG.INF_RESIZE_VALUE * max_aspectratio
            max_width_rangscaling = round(max_width_rangscaling)
            if cfg.EVAL_CROP_SIZE[
                    0] >= max_width_rangscaling and cfg.EVAL_CROP_SIZE[
                        1] >= cfg.AUG.INF_RESIZE_VALUE:
                logger.info(correct_print("EVAL_CROP_SIZE check"))
                logger.info(
                    "satisfy current EVAL_CROP_SIZE: ({},{}) >= ({},{}) ".
                    format(cfg.EVAL_CROP_SIZE[0], cfg.EVAL_CROP_SIZE[1],
                           max_height_rangscaling, cfg.AUG.INF_RESIZE_VALUE))
            else:
                logger.info(error_print("EVAL_CROP_SIZE check"))
                logger.info(
                    "EVAL_CROP_SIZE must >= img size({},{}), current EVAL_CROP_SIZE is ({},{})"
                    .format(max_width_rangscaling, cfg.AUG.INF_RESIZE_VALUE,
                            cfg.EVAL_CROP_SIZE[0], cfg.EVAL_CROP_SIZE[1]))
    elif cfg.AUG.AUG_METHOD == "unpadding":
        if len(cfg.AUG.FIX_RESIZE_SIZE) != 2:
            logger.info(error_print("EVAL_CROP_SIZE check"))
            logger.info(
                "you set AUG.AUG_METHOD = 'unpadding', but AUG.FIX_RESIZE_SIZE is wrong. "
                "AUG.FIX_RESIZE_SIZE should be a tuple of length 2")
        elif cfg.EVAL_CROP_SIZE[0] >= cfg.AUG.FIX_RESIZE_SIZE[0] \
                and cfg.EVAL_CROP_SIZE[1] >= cfg.AUG.FIX_RESIZE_SIZE[1]:
            logger.info(correct_print("EVAL_CROP_SIZE check"))
            logger.info(
                "satisfy current EVAL_CROP_SIZE: ({},{}) >= AUG.FIX_RESIZE_SIZE: ({},{}) "
                .format(cfg.EVAL_CROP_SIZE[0], cfg.EVAL_CROP_SIZE[1],
                        cfg.AUG.FIX_RESIZE_SIZE[0], cfg.AUG.FIX_RESIZE_SIZE[1]))
        else:
            logger.info(error_print("EVAL_CROP_SIZE check"))
            logger.info(
                "EVAL_CROP_SIZE: ({},{}) must >= AUG.FIX_RESIZE_SIZE: ({},{})".
                format(cfg.EVAL_CROP_SIZE[0], cfg.EVAL_CROP_SIZE[1],
                       cfg.AUG.FIX_RESIZE_SIZE[0], cfg.AUG.FIX_RESIZE_SIZE[1]))
    else:
        logger.info(
            "\nERROR! cfg.AUG.AUG_METHOD setting wrong, it should be one of "
            "[unpadding, stepscaling, rangescaling]")


def inf_resize_value_check():
    if cfg.AUG.AUG_METHOD == "rangescaling":
        if cfg.AUG.INF_RESIZE_VALUE < cfg.AUG.MIN_RESIZE_VALUE or \
                cfg.AUG.INF_RESIZE_VALUE > cfg.AUG.MIN_RESIZE_VALUE:
            logger.info(
                "\nWARNING! you set AUG.AUG_METHOD = 'rangescaling'"
                "AUG.INF_RESIZE_VALUE: {} not in [AUG.MIN_RESIZE_VALUE, AUG.MAX_RESIZE_VALUE]: "
                "[{}, {}].".format(cfg.AUG.INF_RESIZE_VALUE,
                                   cfg.AUG.MIN_RESIZE_VALUE,
                                   cfg.AUG.MAX_RESIZE_VALUE))


def image_type_check(img_dim):
    """
    验证图像的格式与DATASET.IMAGE_TYPE是否一致
    param
        img_dim: 图像包含的通道数
    return
    """
    if (1 in img_dim or 3 in img_dim) and cfg.DATASET.IMAGE_TYPE == 'rgba':
        logger.info(error_print("DATASET.IMAGE_TYPE check"))
        logger.info("DATASET.IMAGE_TYPE is {} but the type of image has "
                    "gray or rgb\n".format(cfg.DATASET.IMAGE_TYPE))
    elif (1 not in img_dim and 3 not in img_dim
          and 4 in img_dim) and cfg.DATASET.IMAGE_TYPE == 'rgb':
        logger.info(correct_print("DATASET.IMAGE_TYPE check"))
        logger.info(
            "\nWARNING: DATASET.IMAGE_TYPE is {} but the type of all image is rgba"
            .format(cfg.DATASET.IMAGE_TYPE))
    else:
        logger.info(correct_print("DATASET.IMAGE_TYPE check"))


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
        logger.info("The following list is not separated by {}".format(
            cfg.DATASET.SEPARATOR))
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


def label_gray_check():
    if len(label_gray_wrong) == 0:
        logger.info(correct_print("label gray check"))
        logger.info("All label images are gray")
    else:
        logger.info(error_print("label gray check"))
        logger.info(
            "{} label images are not gray\nLabel pixel statistics may be insignificant"
            .format(len(label_gray_wrong)))
        for i in label_gray_wrong:
            logger.debug(i)


def max_img_size_statistics():
    logger.info("\nDoing max image size statistics:")
    logger.info("max width and max height of images are ({},{})".format(
        max_width, max_height))


def num_classes_loss_matching_check():
    loss_type = cfg.SOLVER.LOSS
    num_classes = cfg.DATASET.NUM_CLASSES
    if num_classes > 2 and (("dice_loss" in loss_type) or
                            ("bce_loss" in loss_type)):
        logger.info(
            error_print(
                "loss check."
                " Dice loss and bce loss is only applicable to binary classfication"
            ))
    else:
        logger.info(correct_print("loss check"))


def check_train_dataset():
    list_file = cfg.DATASET.TRAIN_FILE_LIST
    logger.info("-----------------------------\n1. Check train dataset...")
    with open(list_file, 'r') as fid:
        lines = fid.readlines()
        for line in tqdm(lines):
            line = line.strip()
            parts = line.split(cfg.DATASET.SEPARATOR)
            if len(parts) != 2:
                list_wrong.append(line)
                continue
            img_name, grt_name = parts[0], parts[1]
            img_path = os.path.join(cfg.DATASET.DATA_DIR, img_name)
            grt_path = os.path.join(cfg.DATASET.DATA_DIR, grt_name)
            try:
                img = cv2_imread(img_path, cv2.IMREAD_UNCHANGED)
                grt = pil_imread(grt_path)
            except Exception as e:
                imread_failed.append((line, str(e)))
                continue

            is_gray = is_label_gray(grt)
            if not is_gray:
                label_gray_wrong.append(line)
                grt = cv2.cvtColor(grt, cv2.COLOR_BGR2GRAY)
            get_image_max_height_width(img)
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
        label_gray_check()
        gt_check()
        image_type_check(img_dim)
        max_img_size_statistics()
        shape_check()
        num_classes_loss_matching_check()


def check_val_dataset():
    list_file = cfg.DATASET.VAL_FILE_LIST
    logger.info("\n-----------------------------\n2. Check val dataset...")
    with open(list_file) as fid:
        lines = fid.readlines()
        for line in tqdm(lines):
            line = line.strip()
            parts = line.split(cfg.DATASET.SEPARATOR)
            if len(parts) != 2:
                list_wrong.append(line)
                continue
            img_name, grt_name = parts[0], parts[1]
            img_path = os.path.join(cfg.DATASET.DATA_DIR, img_name)
            grt_path = os.path.join(cfg.DATASET.DATA_DIR, grt_name)
            try:
                img = cv2_imread(img_path, cv2.IMREAD_UNCHANGED)
                grt = pil_imread(grt_path)
            except Exception as e:
                imread_failed.append((line, str(e)))
                continue

            is_gray = is_label_gray(grt)
            if not is_gray:
                label_gray_wrong.append(line)
                grt = cv2.cvtColor(grt, cv2.COLOR_BGR2GRAY)
            get_image_max_height_width(img)
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
        label_gray_check()
        gt_check()
        image_type_check(img_dim)
        max_img_size_statistics()
        shape_check()
        eval_crop_size_check(max_height, max_width, min_aspectratio,
                             max_aspectratio)


def check_test_dataset():
    list_file = cfg.DATASET.TEST_FILE_LIST
    has_label = False
    with open(list_file) as fid:
        logger.info("\n-----------------------------\n3. Check test dataset...")
        lines = fid.readlines()
        for line in tqdm(lines):
            line = line.strip()
            parts = line.split(cfg.DATASET.SEPARATOR)
            if len(parts) == 1:
                img_name = parts
                img_path = os.path.join(cfg.DATASET.DATA_DIR, img_name[0])
                try:
                    img = cv2_imread(img_path, cv2.IMREAD_UNCHANGED)
                except Exception as e:
                    imread_failed.append((line, str(e)))
                    continue
            elif len(parts) == 2:
                has_label = True
                img_name, grt_name = parts[0], parts[1]
                img_path = os.path.join(cfg.DATASET.DATA_DIR, img_name)
                grt_path = os.path.join(cfg.DATASET.DATA_DIR, grt_name)
                try:
                    img = cv2_imread(img_path, cv2.IMREAD_UNCHANGED)
                    grt = pil_imread(grt_path)
                except Exception as e:
                    imread_failed.append((line, str(e)))
                    continue

                is_gray = is_label_gray(grt)
                if not is_gray:
                    label_gray_wrong.append(line)
                    grt = cv2.cvtColor(grt, cv2.COLOR_BGR2GRAY)
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
            get_image_max_height_width(img)
            get_image_min_max_aspectratio(img)
            get_image_dim(img)

        file_list_check(list_file)
        imread_check()
        if has_label:
            label_gray_check()
        if has_label:
            gt_check()
        image_type_check(img_dim)
        max_img_size_statistics()
        if has_label:
            shape_check()
        eval_crop_size_check(max_height, max_width, min_aspectratio,
                             max_aspectratio)


def main(args):
    if args.cfg_file is not None:
        cfg.update_from_file(args.cfg_file)
    cfg.check_and_infer()
    logger.info(pprint.pformat(cfg))

    init_global_variable()
    check_train_dataset()

    init_global_variable()
    check_val_dataset()

    init_global_variable()
    check_test_dataset()

    inf_resize_value_check()

    print("\nDetailed error information can be viewed in detail.log file.")


if __name__ == "__main__":
    args = parse_args()
    logger = logging.getLogger()
    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(message)s"
    formatter = logging.Formatter(BASIC_FORMAT)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel('INFO')
    th = logging.FileHandler('detail.log', 'w')
    th.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(th)
    main(args)

# coding: utf8

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

from utils.config import cfg


def init_global_variable():
    """
    初始化全局变量
    """
    global png_format_right_num  # 格式错误的标签图数量
    global png_format_wrong_num  # 格式错误的标签图数量
    global total_grt_classes  # 总的标签类别
    global total_num_of_each_class  # 每个类别总的像素数
    global shape_unequal  # 图片和标签shape不一致
    global png_format_wrong  # 标签格式错误

    png_format_right_num = 0
    png_format_wrong_num = 0
    total_grt_classes = []
    total_num_of_each_class = []
    shape_unequal = []
    png_format_wrong = []


def parse_args():
    parser = argparse.ArgumentParser(description='PaddleSeg check')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default=None,
        type=str)
    return parser.parse_args()


def cv2_imread(file_path, flag=cv2.IMREAD_COLOR):
    # resolve cv2.imread open Chinese file path issues on Windows Platform.
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flag)


def get_image_max_height_width(img, max_height, max_width):
    img_shape = img.shape
    height, width = img_shape[0], img_shape[1]
    max_height = max(height, max_height)
    max_width = max(width, max_width)
    return max_height, max_width


def get_image_min_max_aspectratio(img, min_aspectratio, max_aspectratio):
    img_shape = img.shape
    height, width = img_shape[0], img_shape[1]
    min_aspectratio = min(width / height, min_aspectratio)
    max_aspectratio = max(width / height, max_aspectratio)
    return min_aspectratio, max_aspectratio


def get_image_dim(img, img_dim):
    """获取图像的维度"""
    img_shape = img.shape
    if img_shape[-1] not in img_dim:
        img_dim.append(img_shape[-1])


def sum_gt_check(png_format, grt_classes, num_of_each_class):
    """
    统计所有标签图上的格式、类别和每个类别的像素数
    params:
        png_format: 返回是否是png格式图片
        grt_classes: 标签类别
        num_of_each_class: 各个类别的像素数目
    """
    global png_format_right_num, png_format_wrong_num, total_grt_classes, total_num_of_each_class

    if png_format:
        png_format_right_num += 1
    else:
        png_format_wrong_num += 1

    if cfg.DATASET.IGNORE_INDEX in grt_classes:
        grt_classes2 = np.delete(
            grt_classes, np.where(grt_classes == cfg.DATASET.IGNORE_INDEX))
    if min(grt_classes2) < 0 or max(grt_classes2) > cfg.DATASET.NUM_CLASSES - 1:
        print("fatal error: label class is out of range [0, {}]".format(
            cfg.DATASET.NUM_CLASSES - 1))

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


def gt_check():
    """
    对标签进行校验，输出校验结果
    params：
         png_format_wrong_num： 格式错误的标签图数量
         png_format_right_num:  格式正确的标签图数量
         total_grt_classes： 总的标签类别
         total_num_of_each_class： 每个类别总的像素数目
    return：
        total_nc： 按升序排序后的总标签类别和像素数目
    """
    if png_format_wrong_num == 0:
        print("Not pass label png format check!")
    else:
        print("Pass label png format check!")
    print(
        "total {} label imgs are png format, {} label imgs are not png fromat".
        format(png_format_right_num, png_format_wrong_num))

    total_nc = sorted(zip(total_grt_classes, total_num_of_each_class))
    print("total label calsses and their corresponding numbers:\n{} ".format(
        total_nc))
    if total_nc[0][0]:
        print(
            "Not pass label class check!\nWarning: label classes should start from 0 !!!"
        )
    else:
        print("Pass label class check!")


def ground_truth_check(grt, grt_path):
    """
    验证标签是否重零开始，标签值为0，1，...，num_classes-1, ingnore_idx
    验证标签图像的格式
    返回标签的像素数
    检查图像是否都是ignore_index
    params:
        grt: 标签图
        grt_path: 标签图路径
    return:
        png_format: 返回是否是png格式图片
        label_correct: 返回标签是否是正确的
        label_pixel_num: 返回标签的像素数
    """
    if imghdr.what(grt_path) == "png":
        png_format = True
    else:
        png_format = False

    unique, counts = np.unique(grt, return_counts=True)

    return png_format, unique, counts


def eval_crop_size_check(max_height, max_width, min_aspectratio,
                         max_aspectratio):
    """
    判断eval_crop_siz与验证集及测试集的max_height, max_width的关系
    param
        max_height: 数据集的最大高
        max_width: 数据集的最大宽
    """
    if cfg.AUG.AUG_METHOD == "stepscaling":
        flag = True
        if max_width > cfg.EVAL_CROP_SIZE[0]:
            print(
                "ERROR: The EVAL_CROP_SIZE[0]: {} should larger max width of images {}!"
                .format(cfg.EVAL_CROP_SIZE[0], max_width))
            flag = False
        if max_height > cfg.EVAL_CROP_SIZE[1]:
            print(
                "ERROR: The EVAL_CROP_SIZE[1]: {} should larger max height of images {}!"
                .format(cfg.EVAL_CROP_SIZE[1], max_height))
            flag = False
        if flag:
            print("EVAL_CROP_SIZE setting correct")
    elif cfg.AUG.AUG_METHOD == "rangescaling":
        if min_aspectratio <= 1 and max_aspectratio >= 1:
            if cfg.EVAL_CROP_SIZE[
                    0] >= cfg.AUG.INF_RESIZE_VALUE and cfg.EVAL_CROP_SIZE[
                        1] >= cfg.AUG.INF_RESIZE_VALUE:
                print("EVAL_CROP_SIZE setting correct")
            else:
                print(
                    "ERROR: EVAL_CROP_SIZE: ({},{}) must large than img size({},{})"
                    .format(cfg.EVAL_CROP_SIZE[0], cfg.EVAL_CROP_SIZE[1],
                            cfg.AUG.INF_RESIZE_VALUE, cfg.AUG.INF_RESIZE_VALUE))
        elif min_aspectratio > 1:
            max_height_rangscaling = cfg.AUG.INF_RESIZE_VALUE / min_aspectratio
            max_height_rangscaling = round(max_height_rangscaling)
            if cfg.EVAL_CROP_SIZE[
                    0] >= cfg.AUG.INF_RESIZE_VALUE and cfg.EVAL_CROP_SIZE[
                        1] >= max_height_rangscaling:
                print("EVAL_CROP_SIZE setting correct")
            else:
                print(
                    "ERROR: EVAL_CROP_SIZE: ({},{}) must large than img size({},{})"
                    .format(cfg.EVAL_CROP_SIZE[0], cfg.EVAL_CROP_SIZE[1],
                            cfg.AUG.INF_RESIZE_VALUE, max_height_rangscaling))
        elif max_aspectratio < 1:
            max_width_rangscaling = cfg.AUG.INF_RESIZE_VALUE * max_aspectratio
            max_width_rangscaling = round(max_width_rangscaling)
            if cfg.EVAL_CROP_SIZE[
                    0] >= max_width_rangscaling and cfg.EVAL_CROP_SIZE[
                        1] >= cfg.AUG.INF_RESIZE_VALUE:
                print("EVAL_CROP_SIZE setting correct")
            else:
                print(
                    "ERROR: EVAL_CROP_SIZE: ({},{}) must large than img size({},{})"
                    .format(cfg.EVAL_CROP_SIZE[0], cfg.EVAL_CROP_SIZE[1],
                            max_width_rangscaling, cfg.AUG.INF_RESIZE_VALUE))
    elif cfg.AUG.AUG_METHOD == "unpadding":
        if cfg.EVAL_CROP_SIZE[0] >= cfg.AUG.FIX_RESIZE_SIZE[
                0] and cfg.EVAL_CROP_SIZE[1] >= cfg.AUG.FIX_RESIZE_SIZE[1]:
            print("EVAL_CROP_SIZE setting correct")
        else:
            print(
                "ERROR: EVAL_CROP_SIZE: ({},{}) must large than img size({},{})"
                .format(cfg.EVAL_CROP_SIZE[0], cfg.EVAL_CROP_SIZE[1],
                        cfg.AUG.FIX_RESIZE_SIZE[0], cfg.AUG.FIX_RESIZE_SIZE[1]))
    else:
        print(
            "ERROR: cfg.AUG.AUG_METHOD setting wrong, it should be one of [unpadding, stepscaling, rangescaling]"
        )


def inf_resize_value_check():
    if cfg.AUG.AUG_METHOD == "rangescaling":
        if cfg.AUG.INF_RESIZE_VALUE < cfg.AUG.MIN_RESIZE_VALUE or \
                cfg.AUG.INF_RESIZE_VALUE > cfg.AUG.MIN_RESIZE_VALUE:
            print(
                "ERROR: you set AUG.AUG_METHOD = 'rangescaling'"
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
        print(
            "ERROR: DATASET.IMAGE_TYPE is {} but the type of image has gray or rgb\n"
            .format(cfg.DATASET.IMAGE_TYPE))
    # elif (1 not in img_dim and 3 not in img_dim and 4 in img_dim) and cfg.DATASET.IMAGE_TYPE == 'rgb':
    #     print("ERROR: DATASET.IMAGE_TYPE is {} but the type of image is rgba\n".format(cfg.DATASET.IMAGE_TYPE))
    else:
        print("DATASET.IMAGE_TYPE setting correct")


def image_label_shape_check(img, grt):
    """
    验证图像和标签的大小是否匹配
    """

    flag = True
    img_height = img.shape[0]
    img_width = img.shape[1]
    grt_height = grt.shape[0]
    grt_width = grt.shape[1]

    if img_height != grt_height or img_width != grt_width:
        flag = False
    return flag


def check_train_dataset():
    train_list = cfg.DATASET.TRAIN_FILE_LIST
    print("\ncheck train dataset...")
    with open(train_list, 'r') as fid:
        img_dim = []
        lines = fid.readlines()
        for line in tqdm(lines):
            parts = line.strip().split(cfg.DATASET.SEPARATOR)
            if len(parts) != 2:
                print(
                    line, "File list format incorrect! It should be"
                    " image_name{}label_name\\n ".format(cfg.DATASET.SEPARATOR))
                continue
            img_name, grt_name = parts[0], parts[1]
            img_path = os.path.join(cfg.DATASET.DATA_DIR, img_name)
            grt_path = os.path.join(cfg.DATASET.DATA_DIR, grt_name)
            img = cv2_imread(img_path, cv2.IMREAD_UNCHANGED)
            grt = cv2_imread(grt_path, cv2.IMREAD_GRAYSCALE)

            get_image_dim(img, img_dim)
            is_equal_img_grt_shape = image_label_shape_check(img, grt)
            if not is_equal_img_grt_shape:
                print(line,
                      "ERROR: source img and label img must has the same size")

            png_format, grt_classes, num_of_each_class = ground_truth_check(
                grt, grt_path)
            sum_gt_check(png_format, grt_classes, num_of_each_class)

        gt_check()

        image_type_check(img_dim)


def check_val_dataset():
    val_list = cfg.DATASET.VAL_FILE_LIST
    with open(val_list) as fid:
        max_height = 0
        max_width = 0
        min_aspectratio = sys.float_info.max
        max_aspectratio = 0.0
        img_dim = []
        print("check val dataset...")
        lines = fid.readlines()
        for line in tqdm(lines):
            parts = line.strip().split(cfg.DATASET.SEPARATOR)
            if len(parts) != 2:
                print(
                    line, "File list format incorrect! It should be"
                    " image_name{}label_name\\n ".format(cfg.DATASET.SEPARATOR))
                continue
            img_name, grt_name = parts[0], parts[1]
            img_path = os.path.join(cfg.DATASET.DATA_DIR, img_name)
            grt_path = os.path.join(cfg.DATASET.DATA_DIR, grt_name)
            img = cv2_imread(img_path, cv2.IMREAD_UNCHANGED)
            grt = cv2_imread(grt_path, cv2.IMREAD_GRAYSCALE)

            max_height, max_width = get_image_max_height_width(
                img, max_height, max_width)
            min_aspectratio, max_aspectratio = get_image_min_max_aspectratio(
                img, min_aspectratio, max_aspectratio)
            get_image_dim(img, img_dim)
            is_equal_img_grt_shape = image_label_shape_check(img, grt)
            if not is_equal_img_grt_shape:
                print(line,
                      "ERROR: source img and label img must has the same size")

            png_format, grt_classes, num_of_each_class = ground_truth_check(
                grt, grt_path)
            sum_gt_check(png_format, grt_classes, num_of_each_class)
        gt_check()

        eval_crop_size_check(max_height, max_width, min_aspectratio,
                             max_aspectratio)
        image_type_check(img_dim)


def check_test_dataset():
    test_list = cfg.DATASET.TEST_FILE_LIST
    with open(test_list) as fid:
        max_height = 0
        max_width = 0
        min_aspectratio = sys.float_info.max
        max_aspectratio = 0.0
        img_dim = []
        print("check test dataset...")
        lines = fid.readlines()
        for line in tqdm(lines):
            parts = line.strip().split(cfg.DATASET.SEPARATOR)
            if len(parts) == 1:
                img_name = parts
                img_path = os.path.join(cfg.DATASET.DATA_DIR, img_name)
                img = cv2_imread(img_path, cv2.IMREAD_UNCHANGED)

            elif len(parts) == 2:
                img_name, grt_name = parts[0], parts[1]
                img_path = os.path.join(cfg.DATASET.DATA_DIR, img_name)
                grt_path = os.path.join(cfg.DATASET.DATA_DIR, grt_name)
                img = cv2_imread(img_path, cv2.IMREAD_UNCHANGED)
                grt = cv2_imread(grt_path, cv2.IMREAD_GRAYSCALE)
                is_equal_img_grt_shape = image_label_shape_check(img, grt)
                if not is_equal_img_grt_shape:
                    print(
                        line,
                        "ERROR: source img and label img must has the same size"
                    )

                png_format, grt_classes, num_of_each_class = ground_truth_check(
                    grt, grt_path)
                sum_gt_check(png_format, grt_classes, num_of_each_class)

            else:
                print(
                    line, "File list format incorrect! It should be"
                    " image_name{}label_name\\n or image_name\n ".format(
                        cfg.DATASET.SEPARATOR))
                continue

            max_height, max_width = get_image_max_height_width(
                img, max_height, max_width)
            min_aspectratio, max_aspectratio = get_image_min_max_aspectratio(
                img, min_aspectratio, max_aspectratio)
            get_image_dim(img, img_dim)

        gt_check()
        eval_crop_size_check(max_height, max_width, min_aspectratio,
                             max_aspectratio)
        image_type_check(img_dim)


def main(args):
    if args.cfg_file is not None:
        cfg.update_from_file(args.cfg_file)
    cfg.check_and_infer(reset_dataset=True)
    print(pprint.pformat(cfg))

    init_global_variable()
    check_train_dataset()

    init_global_variable()
    check_val_dataset()

    init_global_variable()
    check_test_dataset()

    inf_resize_value_check()


if __name__ == "__main__":
    args = parse_args()
    args.cfg_file = "../configs/cityscape.yaml"
    main(args)

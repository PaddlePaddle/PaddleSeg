import sys
import cv2
from PIL import Image
import numpy as np
import math
import requests
import multiprocessing
import json
import os
import base64
import cv2

# ###### common configs ######

# pre resize
pre_height = None
pre_width = None
# final output size
target_height = 1250
target_width = 1250
# stride
height_stride = 1250
width_stride = 1250
# padding, always the same as ignore pixel
padding_pixel = 0
# url
url = "http://10.255.94.19:8000/put_image"

# ###########################


def get_color_map_list(num_classes):
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j, lab = 0, i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    return color_map


def compose_img(im_list, rows, cols, save_file):
    image = Image.new('P',
                      (cols * target_width, rows * target_height))  # 创建一个新图
    for y in range(rows):
        for x in range(cols):
            patch = Image.fromarray(im_list[cols * y + x])
            image.paste(patch, (x * target_width, y * target_height))

    color_map = get_color_map_list(256)
    image.putpalette(color_map)
    image.save(save_file)


def divide_img(img):
    src_im_height = img.shape[0]
    src_im_width = img.shape[1]

    ret_imgs = []

    x1, y1, idx = 0, 0, 0
    while y1 < src_im_height:
        y2 = y1 + target_height
        while x1 < src_im_width:
            x2 = x1 + target_width
            img_crop = img[y1:y2, x1:x2]
            if y2 > src_im_height or x2 > src_im_width:
                pad_bottom = y2 - src_im_height if y2 > src_im_height else 0
                pad_right = x2 - src_im_width if x2 > src_im_width else 0
                img_crop = cv2.copyMakeBorder(
                    img_crop,
                    0,
                    pad_bottom,
                    0,
                    pad_right,
                    cv2.BORDER_CONSTANT,
                    value=padding_pixel)
            ret_imgs.append(img_crop)
            x1 += width_stride
            idx += 1
        x1 = 0
        y1 += height_stride

    return ret_imgs


def encode(img):
    img_str = base64.b64encode(cv2.imencode('.png', img)[1]).decode()
    return img_str


def decode(img_str, color=True):
    img_byte = base64.b64decode(img_str)
    img_np_arr = np.fromstring(img_byte, np.uint8)
    if color:
        img = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imdecode(img_np_arr, cv2.IMREAD_GRAYSCALE)
    return img


def send(img1, img2):
    if img2 is not None:
        data = {
            'img1': encode(img1),
            'img2': encode(img2),
        }
    else:
        data = {'img1': encode(img1)}
    return requests.post(url, data=json.dumps(data)).content


def send_request(img1, img2, idx):  #, res1_dict, res2_dict, diff_dict):
    ret = json.loads(send(img1, img2))
    res1 = decode(ret['res_map1'], color=False)
    res1_dict[idx] = res1
    if img2 is not None:
        res2 = decode(ret['res_map2'], color=False)
        diff = decode(ret['diff'], color=False)
        res2_dict[idx] = res2
        diff_dict[idx] = diff


def divide_and_infer(img1, img2, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    src_im_height = img1.shape[0]
    src_im_width = img1.shape[1]

    cols = math.ceil(src_im_width / target_width)
    rows = math.ceil(src_im_height / target_height)
    nums = cols * rows

    patch1 = divide_img(img1)
    patch2 = divide_img(img2) if img2 is not None else [None] * nums
    print("divide into %d patch" % nums)

    global res1_dict, res2_dict, diff_dict
    res1_dict, res2_dict, diff_dict = {}, {}, {}
    for i in range(nums):
        send_request(patch1[i], patch2[i], i)

    res1_list = [res1_dict[i] for i in range(nums)]
    compose_img(res1_list, rows, cols, save_dir + "/res1.png")
    if img2 is not None:
        res2_list = [res2_dict[i] for i in range(nums)]
        compose_img(res2_list, rows, cols, save_dir + "/res2.png")
        diff_list = [diff_dict[i] for i in range(nums)]
        compose_img(diff_list, rows, cols, save_dir + "/diff.png")

    # # multiprocess
    # res1_dict = multiprocessing.Manager()
    # res2_dict = multiprocessing.Manager()
    # diff_dict = multiprocessing.Manager()
    #
    # p_num = 4
    # pool = multiprocessing.Pool(p_num)
    # pool.map(send_request,
    #          (patch1, patch2, list(range(nums)), res1_dict, res2_dict, diff_dict))
    #
    # res1_dict = dict(res1_dict)
    # res1_list = [res1_dict[i] for i in range(nums)]
    # compose_img(res1_list, rows, cols, save_dir + "/res1.png")
    #
    # res2_dict = dict(res2_dict)
    # res2_list = [res2_dict[i] for i in range(nums)]
    # compose_img(res2_list, rows, cols, save_dir + "/res2.png")
    #
    # diff_dict = dict(diff_dict)
    # diff_list = [diff_dict[i] for i in range(nums)]
    # compose_img(diff_list, rows, cols, save_dir + "/diff.png")


def main(im1_file, im2_file, save_dir):
    img1 = cv2.imdecode(np.fromfile(im1_file, dtype=np.uint8), cv2.IMREAD_COLOR)
    img2 = None
    if im2_file is not None:
        img2 = cv2.imdecode(
            np.fromfile(im2_file, dtype=np.uint8), cv2.IMREAD_COLOR)

    if pre_height and pre_width:
        img1 = cv2.resize(img1, (pre_height, pre_width))
        if img2 is not None:
            img2 = cv2.resize(img2, (pre_height, pre_width))
    divide_and_infer(img1, img2, save_dir)


if __name__ == '__main__':
    args = sys.argv
    if len(args) == 2:
        im1_file = args[1]
        im2_file = None
    else:
        im1_file = args[1]
        im2_file = args[2]

    main(im1_file, im2_file, save_dir="result")

import os
import sys
import multiprocessing
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import skimage
import gdal
import numpy as np
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

import solaris as sol
from solaris.raster.image import create_multiband_geotiff
from solaris.utils.core import _check_gdf_load

module_path = os.path.abspath(os.path.join('./src/'))
if module_path not in sys.path:
    sys.path.append(module_path)
from solaris.preproc.image import LoadImage, SaveImage, Resize
from sn7_baseline_prep_funcs import map_wrapper, make_geojsons_and_masks

# ###### common configs for divide images ######

# pre resize
pre_height = None  # 3072
pre_width = None  # 3072
# final output size
target_height = 512
target_width = 512
# stride
height_stride = 512
width_stride = 512
# padding, always the same as ignore pixel
padding_pixel = 255

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


def compose_img(divide_img_ls, compose_img_dir, ext=".png"):
    im_list = sorted(divide_img_ls)

    last_file = os.path.split(im_list[-1])[-1]
    file_name = '_'.join(last_file.split('.')[0].split('_')[:-2])
    yy, xx = last_file.split('.')[0].split('_')[-2:]
    rows = int(yy) // height_stride + 1
    cols = int(xx) // width_stride + 1

    image = Image.new('P',
                      (cols * target_width, rows * target_height))  # 创建一个新图
    for y in range(rows):
        for x in range(cols):
            patch = Image.open(im_list[cols * y + x])
            image.paste(patch, (x * target_width, y * target_height))

    color_map = get_color_map_list(256)
    image.putpalette(color_map)
    image.save(os.path.join(compose_img_dir, file_name + ext))


def compose_arr(divide_img_ls, compose_img_dir, ext=".npy"):
    """
    Core function of putting results into one.
    """
    im_list = sorted(divide_img_ls)

    last_file = os.path.split(im_list[-1])[-1]
    file_name = '_'.join(last_file.split('.')[0].split('_')[:-2])
    yy, xx = last_file.split('.')[0].split('_')[-2:]
    rows = int(yy) // height_stride + 1
    cols = int(xx) // width_stride + 1

    image = np.zeros(
        (cols * target_width, rows * target_height), dtype=np.float32) * 255
    for y in range(rows):
        for x in range(cols):
            patch = np.load(im_list[cols * y + x])
            image[y * target_height:(y + 1) * target_height, x *
                  target_width:(x + 1) * target_width] = patch

    np.save(os.path.join(compose_img_dir, file_name + ext), image)


def divide_img(img_file, save_dir='divide_imgs', inter_type=cv2.INTER_LINEAR):
    """
    Core function of dividing images.
    """
    _, filename = os.path.split(img_file)
    basename, ext = os.path.splitext(filename)

    img = np.array(Image.open(img_file))
    if pre_height is not None and pre_width is not None:
        if 1023 in img.shape:
            offset_h = 1 if img.shape[0] == 1023 else 0
            offset_w = 1 if img.shape[1] == 1023 else 0
            img = cv2.copyMakeBorder(
                img, 0, offset_h, 0, offset_w, cv2.BORDER_CONSTANT, value=255)
        img = cv2.resize(img, (pre_height, pre_width), interpolation=inter_type)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    src_im_height = img.shape[0]
    src_im_width = img.shape[1]

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
            save_file = os.path.join(save_dir,
                                     basename + "_%05d_%05d" % (y1, x1) + ext)
            Image.fromarray(img_crop).save(save_file)
            x1 += width_stride
            idx += 1
        x1 = 0
        y1 += height_stride


def divide(root):
    """
    Considering the training speed, we divide the image into small images.
    """
    locas = [os.path.join(root, x) for x in os.listdir(root)]
    for loca in locas:
        if not os.path.isdir(os.path.join(root, loca)):
            continue
        print(loca)
        img_path = os.path.join(loca, "images_masked_3x")
        imgs = [os.path.join(img_path, x) for x in os.listdir(img_path)]
        for img in imgs:
            divide_img(img, os.path.join(loca, "images_masked_3x_divide"))

        grt_path = os.path.join(loca, "masks_3x")
        if not os.path.exists(grt_path):
            continue
        grts = [os.path.join(grt_path, x) for x in os.listdir(grt_path)]
        for grt in grts:
            divide_img(grt, os.path.join(loca, "masks_3x_divide"),
                       cv2.INTER_NEAREST)


def compose(root):
    """
    Because the images are cut into small parts, the output results are also small parts.
    We need to put the output results into a large one.
    """
    dst = root + "_compose"
    if not os.path.exists(dst):
        os.makedirs(dst)
    dic = {}
    img_files = [os.path.join(root, x) for x in os.listdir(root)]
    for img_file in img_files:
        key = '_'.join(img_file.split('/')[-1].split('_')[2:9])
        if key not in dic:
            dic[key] = [img_file]
        else:
            dic[key].append(img_file)

    for k, v in dic.items():
        print(k)
        compose_arr(v, dst)


def enlarge_3x(root):
    """
    Enlarge the original images by 3 times.
    """
    aois = [os.path.join(root, x) for x in os.listdir(root)]
    for aoi in aois:
        if not os.path.isdir(os.path.join(root, aoi)):
            continue
        print("enlarge 3x:", aoi)
        images_masked = os.path.join(aoi, "images_masked")
        img_files = [
            os.path.join(images_masked, x) for x in os.listdir(images_masked)
        ]
        images_masked_3x = os.path.join(aoi, "images_masked_3x")
        if not os.path.exists(images_masked_3x):
            os.makedirs(images_masked_3x)
        for img_file in img_files:
            lo = LoadImage(img_file)
            img = lo.load()
            _, height, width = img.data.shape

            re = Resize(height * 3, width * 3)
            img = re.resize(img, height * 3, width * 3)
            assert img.data.shape[1] == height * 3
            assert img.data.shape[2] == width * 3

            sa = SaveImage(
                img_file.replace("images_masked", "images_masked_3x"))
            sa.transform(img)


def create_label(root, f3x=True):
    """
    Create label according to given json file.
    If f3x is True, it will create label that enlarged 3 times than original size.
    """
    aois = os.listdir(root)

    n_threads = 10
    make_fbc = False

    input_args = []
    for i, aoi in enumerate(aois):
        if not os.path.isdir(os.path.join(root, aoi)):
            continue
        print(i, "aoi:", aoi)
        im_dir = os.path.join(root, aoi,
                              'images_masked_3x/' if f3x else 'images_masked/')
        json_dir = os.path.join(root, aoi, 'labels_match/')
        out_dir_mask = os.path.join(root, aoi, 'masks_3x/' if f3x else 'masks/')
        out_dir_mask_fbc = os.path.join(
            root, aoi, 'masks_fbc_3x/' if f3x else 'masks_fbc/')
        os.makedirs(out_dir_mask, exist_ok=True)
        if make_fbc:
            os.makedirs(out_dir_mask_fbc, exist_ok=True)

        json_files = sorted([
            f for f in os.listdir(os.path.join(json_dir))
            if f.endswith('Buildings.geojson')
            and os.path.exists(os.path.join(json_dir, f))
        ])
        for j, f in enumerate(json_files):
            # print(i, j, f)
            name_root = f.split('.')[0]
            json_path = os.path.join(json_dir, f)
            image_path = os.path.join(im_dir, name_root + '.tif').replace(
                'labels', 'images').replace('_Buildings', '')
            output_path_mask = os.path.join(out_dir_mask, name_root + '.tif')
            if make_fbc:
                output_path_mask_fbc = os.path.join(out_dir_mask_fbc,
                                                    name_root + '.tif')
            else:
                output_path_mask_fbc = None

            if not os.path.exists(output_path_mask):
                input_args.append([
                    make_geojsons_and_masks, name_root, image_path, json_path,
                    output_path_mask, output_path_mask_fbc
                ])

    print("len input_args", len(input_args))
    print("Execute...\n")
    with multiprocessing.Pool(n_threads) as pool:
        pool.map(map_wrapper, input_args)


def create_trainval_list(root):
    """
    Create train list and validation list.
    Aois in val_aois below are chosen to validation aois.
    """
    val_aois = set([
        "L15-0387E-1276N_1549_3087_13", "L15-1276E-1107N_5105_3761_13",
        "L15-1015E-1062N_4061_3941_13", "L15-1615E-1206N_6460_3366_13",
        "L15-1438E-1134N_5753_3655_13", "L15-0632E-0892N_2528_4620_13",
        "L15-0566E-1185N_2265_3451_13", "L15-1200E-0847N_4802_4803_13",
        "L15-1848E-0793N_7394_5018_13", "L15-1690E-1211N_6763_3346_13"
    ])
    fw1 = open("train_list.txt", 'w')
    fw2 = open("val_list.txt", 'w')
    for aoi in os.listdir(root):
        if not os.path.isdir(os.path.join(root, aoi)):
            continue
        img_path = os.path.join(root, aoi, "images_masked_3x_divide")
        grt_path = os.path.join(root, aoi, "masks_3x_divide")
        for grt_file in os.listdir(grt_path):
            img_file = grt_file.replace("_Buildings", '')
            if os.path.isfile(os.path.join(img_path, img_file)):
                if aoi in val_aois:
                    fw2.write(
                        os.path.join(aoi, "images_masked_3x_divide", img_file) +
                        ' ' + os.path.join(aoi, "masks_3x_divide", grt_file) +
                        '\n')
                else:
                    fw1.write(
                        os.path.join(aoi, "images_masked_3x_divide", img_file) +
                        ' ' + os.path.join(aoi, "masks_3x_divide", grt_file) +
                        '\n')
    fw1.close()
    fw2.close()


def create_test_list(root):
    """
    Create test list.
    """
    fw = open("test_list.txt", 'w')
    for aoi in os.listdir(root):
        if not os.path.isdir(os.path.join(root, aoi)):
            continue
        img_path = os.path.join(root, aoi, "images_masked_3x_divide")
        for img_file in os.listdir(img_path):
            fw.write(
                os.path.join(aoi, "images_masked_3x_divide", img_file) +
                " dummy.tif\n")
    fw.close()

import os
import numpy as np
import cv2
from PIL import Image
from paddleseg import utils
import xml.dom.minidom


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def get_image_list(image_path):
    """Get image list"""
    valid_suffix = [
        '.JPEG', '.jpeg', '.JPG', '.jpg', '.BMP', '.bmp', '.PNG', '.png'
    ]
    image_list = []
    image_dir = None
    if os.path.isfile(image_path):
        if os.path.splitext(image_path)[-1] in valid_suffix:
            image_list.append(image_path)
    elif os.path.isdir(image_path):
        image_dir = image_path
        for root, dirs, files in os.walk(image_path):
            for f in files:
                if '.ipynb_checkpoints' in root:
                    continue
                if os.path.splitext(f)[-1] in valid_suffix:
                    image_list.append(os.path.join(root.split('/')[-1], f))
    else:
        raise FileNotFoundError(
            '`--image_path` is not found. it should be an image file or a directory including images'
        )

    if len(image_list) == 0:
        raise RuntimeError('There are not image file in `--image_path`')

    return image_list, image_dir


def refine_pred():
    image_list, image_dir = get_image_list(
        'detection_out/pseudo_color_prediction')
    for ii in image_list:
        name_pred = 'detection_out/pseudo_color_prediction/' + ii
        name_label = 'data/IDD_Detection/Annotations/all/' + ii[:-3] + 'xml'
        pred = np.array(Image.open(name_pred)).astype(np.float32)
        if not os.path.exists(name_label):
            pred_mask = utils.visualize.get_pseudo_color_map(pred)
            pred_saved_path = 'detect_out/pred_refine/' + ii
            mkdir(pred_saved_path)
            pred_mask.save(pred_saved_path)
            continue

        dom = xml.dom.minidom.parse(name_label)
        root = dom.documentElement
        objects = root.getElementsByTagName("object")
        for item in objects:
            name = item.getElementsByTagName("name")[0]
            if name.firstChild.data == 'traffic sign' or name.firstChild.data == 'traffic light':
                print(ii)
                xmin = int(
                    item.getElementsByTagName('bndbox')[0].getElementsByTagName(
                        'xmin')[0].firstChild.data)
                ymin = int(
                    item.getElementsByTagName('bndbox')[0].getElementsByTagName(
                        'ymin')[0].firstChild.data)
                xmax = int(
                    item.getElementsByTagName('bndbox')[0].getElementsByTagName(
                        'xmax')[0].firstChild.data)
                ymax = int(
                    item.getElementsByTagName('bndbox')[0].getElementsByTagName(
                        'ymax')[0].firstChild.data)
                if name.firstChild.data == 'traffic sign':
                    pred[ymin:ymax, xmin:xmax] = 18
                elif name.firstChild.data == 'traffic light':
                    pred[ymin:ymax, xmin:xmax] = 19

        pred_mask = utils.visualize.get_pseudo_color_map(pred)
        pred_saved_path = 'detect_out/pred_refine/' + ii
        mkdir(pred_saved_path)
        pred_mask.save(pred_saved_path)


def test():
    path = '/Users/liliulei/Downloads/IDD_Detection/JPEGImages/frontNear/'
    image_list, image_dir = get_image_list(path)

    for ii in image_list:
        name_xml = '/Users/liliulei/Downloads/IDD_Detection/Annotations/frontNear/' + ii[:
                                                                                         -3] + 'xml'
        image = cv2.imread(path + ii)
        # print(image.shape)
        (h, w) = image.shape[0:2]

        pred = np.zeros_like(image)

        dom = xml.dom.minidom.parse(name_xml)
        root = dom.documentElement
        objects = root.getElementsByTagName("object")
        for item in objects:
            name = item.getElementsByTagName("name")[0]
            print(name.firstChild.data)
            if name.firstChild.data == 'traffic sign' or name.firstChild.data == 'traffic light':
                xmin = int(
                    item.getElementsByTagName('bndbox')[0].getElementsByTagName(
                        'xmin')[0].firstChild.data)
                ymin = int(
                    item.getElementsByTagName('bndbox')[0].getElementsByTagName(
                        'ymin')[0].firstChild.data)
                xmax = int(
                    item.getElementsByTagName('bndbox')[0].getElementsByTagName(
                        'xmax')[0].firstChild.data)
                ymax = int(
                    item.getElementsByTagName('bndbox')[0].getElementsByTagName(
                        'ymax')[0].firstChild.data)
                if name.firstChild.data == 'traffic sign':
                    pred[ymin:ymax, xmin:xmax, 0] = 255
                elif name.firstChild.data == 'traffic light':
                    pred[ymin:ymax, xmin:xmax, 1] = 255

        new_im = image * 0.5 + pred * 0.5

        cv2.imwrite(ii.split('/')[-1][:-3] + 'png', new_im)


refine_pred()

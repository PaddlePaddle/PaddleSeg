# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os.path as osp
import shutil
import random
from tqdm import tqdm
from qtpy import QtGui
import argparse

# 参考paddlex数据准备文档
# https://github.com/PaddlePaddle/PaddleX/blob/release/2.0.0/docs/data/format/README.md

# 支持的图像格式
FORMATS = [
    ".{}".format(fmt.data().decode())
    for fmt in QtGui.QImageReader.supportedImageFormats()
]
FORMATS.append(".dcm")

# print(FORMATS)


def mkdirp(path):
    if not osp.exists(path):
        os.mkdir(path)


# 语义分割
def Eiseg2Semantic(save_folder, imgs_folder, lab_folder=None, split_rate=0.9):
    """Convert the data marked by eiseg into the semantic segmentation data of paddlex.

    Args:
        save_folder (str): Data save folder.
        imgs_folder (str): Image storage folder.
        lab_folder (str, optional): Label storage folder, 
            if it is none, it will be saved in the current folder by default. Defaults to None.
        split_rate (float, optional): Proportion of training data and validation data. Defaults to 0.9.
    """
    # move
    save_img_folder = osp.join(save_folder, "JPEGImages")
    save_lab_folder = osp.join(save_folder, "Annotations")
    mkdirp(save_folder)
    mkdirp(save_img_folder)
    mkdirp(save_lab_folder)
    imgs_name = os.listdir(imgs_folder)
    if lab_folder is None:
        lab_folder = osp.join(imgs_folder, "label")
    for name in tqdm(imgs_name):
        ext = "." + name.split(".")[-1]
        if ext.lower() in FORMATS:
            img_path = osp.join(imgs_folder, name)
            lab_path = osp.join(lab_folder, name.replace(ext, ".png"))
            if not osp.exists(lab_path):
                lab_path = osp.join(lab_folder, name)
            save_img_path = osp.join(save_img_folder, name)
            save_lab_path = osp.join(save_lab_folder,
                                     os.path.split(lab_path)[-1])
            if osp.exists(img_path) and osp.exists(lab_path):
                shutil.copy(img_path, save_img_path)
                shutil.copy(lab_path, save_lab_path)
    print("===== copy data finished! =====")
    # create label
    label_path = osp.join(lab_folder, "autosave_label.txt")
    save_label_path = osp.join(save_folder, "labels.txt")
    with open(label_path, "r") as rf:
        with open(save_label_path, "w") as wf:
            tmps = rf.readlines()
            wf.write("background\n")
            for i in range(len(tmps)):
                lab = tmps[i].split(" ")[1]
                wf.write(lab + "\n")
    print("===== create label finished! =====")
    # create list
    train_list_path = osp.join(save_folder, "train_list.txt")
    eval_list_path = osp.join(save_folder, "val_list.txt")
    new_imgs_name = os.listdir(save_img_folder)
    random.shuffle(new_imgs_name)
    lens = len(new_imgs_name)
    with open(train_list_path, "w") as tf:
        with open(eval_list_path, "w") as ef:
            for idx, name in tqdm(enumerate(new_imgs_name, start=1)):
                new_img_path = osp.join("JPEGImages", name)
                ext = "." + name.split(".")[-1]
                new_lab_path = osp.join("Annotations",
                                        name.replace(ext, ".png"))
                if not osp.exists(osp.join(save_folder, new_lab_path)):
                    new_lab_path = osp.join("Annotations", name)
                new_img_path = new_img_path.replace("\\", "/")
                new_lab_path = new_lab_path.replace("\\", "/")
                if (idx / lens) <= split_rate:
                    tf.write(new_img_path + " " + new_lab_path + "\n")
                else:
                    ef.write(new_img_path + " " + new_lab_path + "\n")
    print("===== create data list finished! =====")
    print("===== all done! =====")


parser = argparse.ArgumentParser(
    description='Save path, image path, label path and split rate')
parser.add_argument('--save_path', '-d', help='保存文件夹路径，必要参数', required=True)
parser.add_argument('--image_path', '-o', help='图像文件夹路径，必要参数', required=True)
parser.add_argument('--label_path', '-l', help='标签文件夹路径', default=None)
parser.add_argument('--split_rate', '-s', help='分割参数', default=0.9)
args = parser.parse_args()

if __name__ == "__main__":
    save_path = args.save_path
    image_path = args.image_path
    label_path = args.label_path
    split_rate = args.split_rate
    Eiseg2Semantic(save_path, image_path, label_path, split_rate)

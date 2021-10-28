import os
import os.path as osp
import shutil
import random
from tqdm import tqdm
from qtpy import QtGui
import argparse


# refer to the paddlex
# https://github.com/PaddlePaddle/PaddleX/blob/release/2.0.0/docs/data/format/README.md

# supported image formats
FORMATS = [".{}".format(fmt.data().decode()) 
            for fmt in QtGui.QImageReader.supportedImageFormats()]
FORMATS.append(".dcm")
# print(FORMATS)

def mkdirp(path):
    if not osp.exists(path):
        os.mkdir(path)


# semantic segmentation
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
            save_lab_path = osp.join(save_lab_folder, os.path.split(lab_path)[-1])
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
    print("===== create labels.txt finished! =====")
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
                new_lab_path = osp.join("Annotations", name.replace(ext, ".png"))
                if not osp.exists(osp.join(save_folder, new_lab_path)):
                    new_lab_path = osp.join("Annotations", name)
                new_img_path = new_img_path.replace("\\", "/")
                new_lab_path = new_lab_path.replace("\\", "/")
                if (idx / lens) <= split_rate:
                    tf.write(new_img_path + " " + new_lab_path + "\n")
                else:
                    ef.write(new_img_path + " " + new_lab_path + "\n")
    print("===== create data_list.txt finished! =====")
    print("===== all done! =====")


parser = argparse.ArgumentParser(description='Save path, image path, label path and split rate')
parser.add_argument('--save_path', '-d', help='Save folder path, required', required=True)
parser.add_argument('--image_path', '-o', help='Image folder path, required', required=True)
parser.add_argument('--label_path', '-l', help='Label folder path, optional', default=None)
parser.add_argument('--split_rate', '-s', help='Proportion of training set and verification set, optional', default=0.9)
args = parser.parse_args()

if __name__ == "__main__":
    save_path = args.save_path
    image_path = args.image_path
    label_path = args.label_path
    split_rate = args.split_rate
    Eiseg2Semantic(save_path, image_path, label_path, split_rate)
import os
import numpy as np
import glob

num_val = 20000  # the number of validation set
dataset_root = "/ssd3/liuyi22/dataset/remote_sensing_image"
img_dir = os.path.join(dataset_root, 'img_train')
label_dir = os.path.join(dataset_root, 'lab_train')

train_file = open(os.path.join(dataset_root, "train.txt"), "w")
val_file = open(os.path.join(dataset_root, "val.txt"), "w")

img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))

total_idx = np.arange(len(img_files))
np.random.shuffle(total_idx)
val_idx = total_idx[:num_val]

for i, img_path in enumerate(img_files):
    img_path = img_path.replace("/ssd3/liuyi22/dataset/remote_sensing_image/",
                                "")
    label_path = img_path.replace("img_train", "lab_train")
    label_path = label_path[:-3] + "png"
    pair_str = img_path + " " + label_path + "\n"
    if i in val_idx:
        val_file.write(pair_str)
    else:
        train_file.write(pair_str)

train_file.close()
val_file.close()

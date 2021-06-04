import os
import random

import cv2


def get_from_pm85(data_path="/mnt/chenguowei01/datasets/matting/PhotoMatte85",
                  save_path="/mnt/chenguowei01/datasets/matting/gather"):
    """
    Get matte from PhotoMatte85
    """

    files = os.listdir(data_path)
    files = [os.path.join(data_path, f) for f in files]
    random.seed(1)
    random.shuffle(files)
    train_files = files[:-10]
    val_files = files[-10:]

    # training dataset
    fg_save_path = os.path.join(save_path, 'fg', 'PhotoMatte85', 'train')
    alpha_save_path = fg_save_path.replace('fg', 'alpha')
    if not os.path.exists(fg_save_path):
        os.makedirs(fg_save_path)
    if not os.path.exists(alpha_save_path):
        os.makedirs(alpha_save_path)
    for f in train_files:
        png_img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        fg = png_img[:, :, :3]
        alpha = png_img[:, :, -1]
        if alpha[0, 0] != 0:
            alpha[:100, :] = 0
            fg[:100, :, :] = 0
        basename = os.path.basename(f)
        cv2.imwrite(os.path.join(fg_save_path, basename), fg)
        cv2.imwrite(os.path.join(alpha_save_path, basename), alpha)

    # val dataset
    fg_save_path = os.path.join(save_path, 'fg', 'PhotoMatte85', 'val')
    alpha_save_path = fg_save_path.replace('fg', 'alpha')
    if not os.path.exists(fg_save_path):
        os.makedirs(fg_save_path)
    if not os.path.exists(alpha_save_path):
        os.makedirs(alpha_save_path)
    for f in val_files:
        png_img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        fg = png_img[:, :, :3]
        alpha = png_img[:, :, -1]
        if alpha[0, 0] != 0:
            alpha[:100, :] = 0
            fg[:100, :, :] = 0
        basename = os.path.basename(f)
        cv2.imwrite(os.path.join(fg_save_path, basename), fg)
        cv2.imwrite(os.path.join(alpha_save_path, basename), alpha)


if __name__ == "__main__":
    get_from_pm85()

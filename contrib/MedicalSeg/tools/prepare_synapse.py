import SimpleITK as sitk
import numpy as np
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Data preprocessing')
    # params of training
    parser.add_argument(
        "--raw_folder",
        dest="raw_folder",
        help="The raw data path.",
        default=r"Synapse\RawData\RawData\Training",
        type=str)
    parser.add_argument(
        "--save_path",
        dest="save_path",
        help="The save path.",
        default=r"Synapse\preprocessed",
        type=str)
    parser.add_argument(
        "--top_val",
        dest="top_val",
        help="The maximum value for clip.",
        default=275,
        type=float)
    parser.add_argument(
        "--bottom_val",
        dest="bottom_val",
        help="The minimum value for clip.",
        default=-125,
        type=float)
    parser.add_argument(
        "--split_val",
        dest="split_val",
        help="The rate of data for val.",
        default=0.3,
        type=float)
    parser.add_argument(
        "--ignore_label",
        dest="ignore_label",
        help="Indices of labels to ignore. Because some labels are not exist in this paper",
        default=[5, 9, 10, 12, 13],
        type=list)
    parser.add_argument(
        "--ori_classes",
        dest="ori_classes",
        help="The num of classes in raw_data",
        default=14,
        type=int)

    return parser.parse_args()


def build_transfer_label(ori_classes, ignore_label):
    transefer_dict = {}
    count = 1
    for label in range(ori_classes):
        if (label in ignore_label) or (label == 0):
            transefer_dict[label] = 0
        else:
            transefer_dict[label] = count
            count = count + 1
    return transefer_dict


def normalize(data_array):
    data_mean = np.mean(data_array)
    data_std = np.std(data_array)
    new_data = (data_array - data_mean) / data_std
    return new_data


def read_data(filename):
    data = sitk.ReadImage(filename)
    array = sitk.GetArrayFromImage(data)
    return array


def preprocess_data(filename, save_path, top, bottom, mode="train"):
    basename = os.path.basename(filename)
    basename = basename.split(".")[0]
    data = read_data(filename)
    data = np.clip(data, bottom, top)
    data = normalize(data)
    if mode == "train":
        num_slices = data.shape[0]
        for i in range(num_slices):
            slice_data = data[i:i + 1]
            save_name = basename + "_slice%03d" % i + ".npy"
            np.save(
                os.path.join(save_path, "train", "img", save_name), slice_data)
    else:
        save_name = basename + '.npy'
        np.save(os.path.join(save_path, "val", "img", save_name), data)


def preprocess_label(transefer_dict, filename, save_path, mode="train"):
    basename = os.path.basename(filename)
    basename = basename.split(".")[0]
    data = read_data(filename)
    for key, value in transefer_dict.items():
        data[data == key] = value
    if mode == "train":
        num_slices = data.shape[0]
        for i in range(num_slices):
            slice_data = data[i:i + 1]
            save_name = basename + "_slice%03d" % i + ".npy"
            np.save(
                os.path.join(save_path, "train", "label", save_name),
                slice_data)
    else:
        save_name = basename + '.npy'
        np.save(os.path.join(save_path, "val", "label", save_name), data)


def main(args):
    raw_folder = args.raw_folder
    save_path = args.save_path
    split_val = args.split_val
    bottom = args.bottom_val
    top = args.top_val
    ignore_label = args.ignore_label
    ori_classes = args.ori_classes
    transefer_dict = build_transfer_label(ori_classes, ignore_label)

    os.makedirs(os.path.join(save_path, "train", 'img'), exist_ok=True)
    os.makedirs(os.path.join(save_path, "train", 'label'), exist_ok=True)
    os.makedirs(os.path.join(save_path, "val", 'img'), exist_ok=True)
    os.makedirs(os.path.join(save_path, "val", 'label'), exist_ok=True)

    filename_list = os.listdir(os.path.join(raw_folder, "img"))
    filename_list.sort()

    nums_val = int(split_val * len(filename_list))

    for filename_train in filename_list[:-nums_val]:
        image_path = os.path.join(raw_folder, 'img', filename_train)
        preprocess_data(image_path, save_path, top, bottom, mode='train')
        label_path = os.path.join(raw_folder, 'label',
                                  filename_train.replace("img", 'label'))
        preprocess_label(transefer_dict, label_path, save_path, mode='train')

    for filename_val in filename_list[-nums_val:]:
        image_path = os.path.join(raw_folder, 'img', filename_val)
        preprocess_data(image_path, save_path, top, bottom, mode='val')
        label_path = os.path.join(raw_folder, 'label',
                                  filename_val.replace("img", 'label'))
        preprocess_label(transefer_dict, label_path, save_path, mode='val')


if __name__ == '__main__':
    args = parse_args()
    main(args)

#    Copyright 2022 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import nibabel as nib
import argparse
import shutil
import numpy as np
from preprocess_utils.file_and_folder_operations import *
from preprocess_utils.geometry import *
from collections import OrderedDict
from sklearn.model_selection import KFold

from tqdm import tqdm





def preprocess_data(raw_data_path, preprocessed_path, new_spacing):
    save_images_path = os.path.join(preprocessed_path, "images")
    save_labels_path = os.path.join(preprocessed_path, "labels")
    maybe_mkdir_p(save_images_path)
    maybe_mkdir_p(save_labels_path)
    data_lists = os.listdir(os.path.join(raw_data_path, "imagesTr"))
    for filename in tqdm(data_lists):
        nimg = nib.load(os.path.join(raw_data_path, "imagesTr", filename))
        nlabel = nib.load(os.path.join(raw_data_path, "labelsTr", filename))
        data_arrary = nimg.get_data()
        label_array = nlabel.get_data()
        original_spacing = nimg.header["pixdim"][1:4]
        assert data_arrary.shape == label_array.shape
        shape = data_arrary.shape
        new_shape = np.round((
            (np.array(original_spacing) / np.array(new_spacing)).astype(float) *
            np.array(shape))).astype(int)
        new_data_array = resize_image(data_arrary, new_shape)
        new_label_array = resize_segmentation(label_array, new_shape)
        #将数据从hwd转化为dhw
        new_data_array = np.transpose(new_data_array, [2, 0, 1])
        new_label_array = np.transpose(new_label_array, [2, 0, 1])
        np.save(
            os.path.join(save_images_path,
                         filename.replace(r".nii.gz", '.npy')), new_data_array)
        np.save(
            os.path.join(save_labels_path,
                         filename.replace(r".nii.gz", '.npy')), new_label_array)


def clean_raw_data(folder, folder_test, out_folder):
    maybe_mkdir_p(join(out_folder, "imagesTr"))
    maybe_mkdir_p(join(out_folder, "imagesTs"))
    maybe_mkdir_p(join(out_folder, "labelsTr"))

    # train
    all_train_files = []
    patient_dirs_train = subfolders(folder, prefix="patient")
    for p in patient_dirs_train:
        current_dir = p
        data_files_train = [
            i for i in subfiles(
                current_dir, suffix=".nii.gz")
            if i.find("_gt") == -1 and i.find("_4d") == -1
        ]
        corresponding_seg_files = [
            i[:-7] + "_gt.nii.gz" for i in data_files_train
        ]
        for d, s in zip(data_files_train, corresponding_seg_files):
            patient_identifier = d.split("/")[-1][:-7]
            all_train_files.append(patient_identifier + "_0000.nii.gz")
            shutil.copy(d,
                        join(out_folder, "imagesTr",
                             patient_identifier + "_0000.nii.gz"))
            shutil.copy(s,
                        join(out_folder, "labelsTr",
                             patient_identifier + "_0000.nii.gz"))

    # test
    all_test_files = []
    patient_dirs_test = subfolders(folder_test, prefix="patient")
    for p in patient_dirs_test:
        current_dir = p
        data_files_test = [
            i for i in subfiles(
                current_dir, suffix=".nii.gz")
            if i.find("_gt") == -1 and i.find("_4d") == -1
        ]
        for d in data_files_test:
            patient_identifier = d.split("/")[-1][:-7]
            all_test_files.append(patient_identifier + "_0000.nii.gz")
            shutil.copy(d,
                        join(out_folder, "imagesTs",
                             patient_identifier + "_0000.nii.gz"))

    json_dict = OrderedDict()
    json_dict['name'] = "ACDC"
    json_dict['description'] = "cardias cine MRI segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see ACDC challenge"
    json_dict['licence'] = "see ACDC challenge"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {"0": "MRI", }
    json_dict['labels'] = {"0": "background", "1": "RV", "2": "MLV", "3": "LVC"}
    json_dict['numTraining'] = len(all_train_files)
    json_dict['numTest'] = len(all_test_files)
    json_dict['training'] = [{
        'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1][:-12],
        "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1][:-12]
    } for i in all_train_files]
    json_dict['test'] = [
        "./imagesTs/%s.nii.gz" % i.split("/")[-1][:-12] for i in all_test_files
    ]

    save_json(json_dict, os.path.join(out_folder, "dataset.json"))

    splits = []
    patients = np.unique([i[:10] for i in all_train_files])
    patientids = [i[:-12] for i in all_train_files]

    kf = KFold(5, shuffle=True, random_state=12345)
    for tr, val in kf.split(patients):
        splits.append(OrderedDict())
        tr_patients = patients[tr]
        splits[-1]['train'] = [
            i[:-7] for i in all_train_files if i[:10] in tr_patients
        ]
        val_patients = patients[val]
        splits[-1]['val'] = [
            i[:-7] for i in all_train_files if i[:10] in val_patients
        ]
    return splits


def parse_args():
    parser = argparse.ArgumentParser(description='Data preprocessing')
    # params of training
    parser.add_argument(
        "--folder",
        dest="folder",
        help="The training data path.",
        default="/home/aistudio/training",
        type=str)
    parser.add_argument(
        "--folder_test",
        dest="folder_test",
        help="The test data path.",
        default="/home/aistudio/testing",
        type=str)
    parser.add_argument(
        "--clean_folder",
        dest="clean_folder",
        help="The output path of cleaned data path.",
        default="data/ACDCDataset/clean_data",
        type=str)
    parser.add_argument(
        "--preprocessed_folder",
        dest="preprocessed_folder",
        help="The output path of preprocessed data path.",
        default="data/ACDCDataset/preprocessed",
        type=str)
    parser.add_argument(
        '--use_k5',
        dest='use_k5',
        help='Whether to generate 5-fold cross validation txt file ',
        action='store_true')
    return parser.parse_args()


def main(args):
    folder = args.folder
    folder_test = args.folder_test
    clean_folder = args.clean_folder
    preprocessed_folder = args.preprocessed_folder
    new_spacing = [1.52, 1.52, 6.35]
    print("start cleaning data....")
    splits = clean_raw_data(folder, folder_test, clean_folder)
    print("start preprocessing.....")
    preprocess_data(clean_folder, preprocessed_folder, new_spacing)
    for i in range(len(splits)):
        txtname = [
            os.path.join(preprocessed_folder, 'train_list_{}.txt'.format(i)),
            os.path.join(preprocessed_folder, 'val_list_{}.txt'.format(i))
        ]

        with open(txtname[0], "w") as f:
            for filename in splits[i]["train"]:
                f.write("images/{}.npy labels/{}.npy\n".format(filename,
                                                               filename))
        with open(txtname[1], "w") as f:
            for filename in splits[i]['val']:

                f.write("images/{}.npy labels/{}.npy\n".format(filename,
                                                               filename))
        if not args.use_k5:
            break


if __name__ == '__main__':
    args = parse_args()
    main(args)

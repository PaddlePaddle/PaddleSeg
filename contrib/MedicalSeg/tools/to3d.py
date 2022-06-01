import os
import os.path as osp
import sys
import zipfile
import functools
import numpy as np
import nrrd
import time
import glob
import argparse
import zipfile
import collections

import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm
import json
import scipy

import sys 
sys.path.append("PaddleSeg/contrib/MedicalSeg/") 

from medicalseg.utils import get_image_list
from tools.preprocess_utils import uncompressor, global_var, add_qform_sform


sys.path.append(osp.join(osp.dirname(osp.realpath(__file__)), ".."))



def zscore(image):
    print(image.shape)
    image=scipy.stats.zscore(image, axis=0, nan_policy='omit')
    return image



sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))




tasks = {
    1: {
        "Task01_BrainTumour.tar":
        "https://bj.bcebos.com/v1/ai-studio-online/netdisk/975fea1d4c8549b883b2b4bb7e6a82de84392a6edd054948b46ced0f117fd701?responseContentDisposition=attachment%3B%20filename%3DTask01_BrainTumour.tar&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-01-21T18%3A50%3A30Z%2F-1%2F%2F283ea6f8700c129903e3278ea38a54eac2cf087e7f65197268739371898aa1b3"
    },  # 4d
    2: {
        "Task02_Heart.tar":
        "https://bj.bcebos.com/v1/ai-studio-online/netdisk/44a1e00baf55489db5d95d79f2e56e7230b6f87687604ab0889e0deb45ba289e?responseContentDisposition=attachment%3B%20filename%3DTask02_Heart.tar&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-01-21T18%3A30%3A22Z%2F-1%2F%2F3c23a084e9bbbc57d8d6435eb014b7fb8c4160395a425bc94da5b55a08fc14de"
    },  # 3d
    3: {
        "Task03_Liver.tar":
        "https://bj.bcebos.com/v1/ai-studio-online/netdisk/e641b1b7f364472c885147b6c500842f559ee6ae03494b78b5d140d53db35907?responseContentDisposition=attachment%3B%20filename%3DTask03_Liver.tar&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-01-21T18%3A49%3A33Z%2F-1%2F%2F83b1b4e70026a2a568dcfbbf60fb06f0ae27a847e7ebe5ba7b2efe60fc6b16a5"
    },  # 3d
    4: {
        "Task04_Hippocampus.tar":
        "https://bj.bcebos.com/v1/ai-studio-online/1bf93142b1284f69a2a2a4e84248a0fe2bdb76c3b4ba4ddf82754e23d8820dfe?responseContentDisposition=attachment%3B%20filename%3DTask04_Hippocampus.tar&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-02-14T17%3A09%3A53Z%2F-1%2F%2Fc53aa0df7f8810277261a00458d0af93df886c354c27498607bb8e2fb64a3d90"
    },  # 3d
    5: {
        "Task05_Prostate.tar":
        "https://bj.bcebos.com/v1/ai-studio-online/netdisk/aca74eceef674a74bff647998413ebf25a33ad44e04643d7b796e05eecbc9891?responseContentDisposition=attachment%3B%20filename%3DTask05_Prostate.tar&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-01-21T18%3A28%3A58Z%2F-1%2F%2F610d78c178a2f5eeb5d8f6c7ec48ef52f7d6899b5ed8484f213ff1e03d266bd8"
    },  # 4d
    6: {
        "Task06_Lung.tar":
        "https://bj.bcebos.com/v1/ai-studio-online/netdisk/c42c621dc5c0490baaec935e1efd899478615f02add040649764c80c5f46805a?responseContentDisposition=attachment%3B%20filename%3DTask06_Lung.tar&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-01-21T18%3A59%3A27Z%2F-1%2F%2Fd4a6b5b382136af96395a8acc6d18d4e88ac744314c517f19f3a71417be3d12c"
    },  # 3d
    7: {
        "Task07_Pancreas.tar":
        "https://bj.bcebos.com/v1/ai-studio-online/netdisk/d94f22313d764d808b15b240da0335a9cf0ca0e806ce418f9213f9db9e56a5a8?responseContentDisposition=attachment%3B%20filename%3DTask07_Pancreas.tar&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-01-21T18%3A34%3A45Z%2F-1%2F%2F3a17fb265c8fcdac91de8f15e7e2352a31783bbb121755ad27c28685ce047afa"
    },  # 3d
    8: {
        "Task08_HepaticVessel.tar":
        "https://bj.bcebos.com/v1/ai-studio-online/netdisk/51ff9421bfa648449f12e65a68862215c6b5b85f91de49aab1c16626c62c3af6?responseContentDisposition=attachment%3B%20filename%3DTask08_HepaticVessel.tar&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-01-21T18%3A35%3A23Z%2F-1%2F%2Fa664645e0b0c99e351f31352701dbe163de3fbe6e96eac11539629b5e6658360"
    },  # 3d
    9: {
        "Task09_Spleen.tar":
        "https://bj.bcebos.com/v1/ai-studio-online/netdisk/c02462f396f14b13a50d2c9ff01f86fc471c7bff8df24994af7bd8b2298dc843?responseContentDisposition=attachment%3B%20filename%3DTask09_Spleen.tar&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-01-21T18%3A45%3A46Z%2F-1%2F%2Faf6f10f658fbe9569eb423fc1b7bd464aead582ef89cd7c135dcae002bc3cb09"
    },  # 3d
    10: {
        "Task10_Colon.tar":
        "https://bj.bcebos.com/v1/ai-studio-online/netdisk/062aa5a52cc44597a87f56c5ef1371c7acb52f73a2c946be9fea347dedec5058?responseContentDisposition=attachment%3B%20filename%3DTask10_Colon.tar&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-01-21T18%3A42%3A03Z%2F-1%2F%2F106546582e748224f0833e100fc74d1bf3ff7fe4f4370d43bb487b10c3f5deae"
    },  # 3d
}

class Prep:
    def __init__(self,
                 dataset_root="data/TemDataSet",
                 raw_dataset_dir="TemDataSet_seg_raw/",
                 images_dir="train_imgs",
                 labels_dir="train_labels",
                 phase_dir="phase0",
                 urls=None,
                 valid_suffix=("nii.gz", "nii.gz"),
                 filter_key=(None, None),
                 uncompress_params={"format": "zip",
                                    "num_files": 1},
                 images_dir_test=""):
        """
        Create proprosessor for medical dataset.
        Folder structure:
            dataset_root
            ├── raw_dataset_dir
            │   ├── image_dir
            │   ├── labels_dir  
            │   ├── images_dir_test       
            ├── phase_dir
            │   ├── images
            │   ├── labels
            │   ├── train_list.txt
            │   └── val_list.txt
            ├── archive_1.zip
            ├── archive_2.zip
            └── ... archives ...
        Args:
            urls (dict): Urls to download dataset archive. Key will be used as archive name.
            valid_suffix(tuple):  Only files with the assigned suffix will be considered. The first is the suffix for image, and the other is for label.
            filter_key(tuple): Only files containing the filter_key the will be considered.
        """
        # combine all paths
        self.dataset_root = dataset_root
        self.phase_path = os.path.join(self.dataset_root, phase_dir)
        self.raw_data_path = os.path.join(self.dataset_root, raw_dataset_dir)
        self.dataset_json_path = os.path.join(
            self.raw_data_path,
            "dataset.json")  # save the dataset.json to raw path
        self.image_path = os.path.join(self.phase_path, "images")
        self.label_path = os.path.join(self.phase_path, "labels")

        os.makedirs(self.dataset_root, exist_ok=True)
        os.makedirs(self.phase_path, exist_ok=True)
        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.label_path, exist_ok=True)
        self.gpu_tag = "GPU" 
        self.urls = urls

        if osp.exists(self.raw_data_path):
            print(
                f"raw_dataset_dir {self.raw_data_path} exists, skipping uncompress. To uncompress again, remove this directory"
            )
        else:
            self.uncompress_file(
                num_files=uncompress_params["num_files"],
                form=uncompress_params["format"])

        # self.image_files_test = None
        # if len(images_dir_test
        #        ) != 0:  # test image filter is the same as training image
        #     self.image_files_test = get_image_list(
        #         os.path.join(self.raw_data_path, images_dir_test),
        #         valid_suffix[0], filter_key[0])
        #     self.image_files_test.sort()
        #     self.image_path_test = os.path.join(self.phase_path, 'images_test')
        #     os.makedirs(self.image_path_test, exist_ok=True)

        # Load the needed file with filter

        self.image_files = get_image_list(os.path.join(self.raw_data_path, images_dir), valid_suffix[0],filter_key[0])
        self.label_files = get_image_list(os.path.join(self.raw_data_path, labels_dir), valid_suffix[1],filter_key[1])


        print(666666666666666666666)
        print(len(self.image_files))
        print(len(self.label_files))
        # self.image_files.sort()
        # self.label_files.sort()




    def uncompress_file(self, num_files, form):
        uncompress_tool = uncompressor(
            download_params=(self.urls, self.dataset_root, True))
        """unzip all the file in the root directory"""
        files = glob.glob(os.path.join(self.dataset_root, "*.{}".format(form)))

        assert len(files) == num_files, "The file directory should include {} compressed files, but there is only {}".format(num_files, len(files))

        for f in files:
            extract_path = os.path.join(self.raw_data_path,
                                        f.split("/")[-1].split('.')[0])
            uncompress_tool._uncompress_file(
                f, extract_path, delete_file=False, print_progress=True)

    @staticmethod
    def load_medical_data(f):
        """
        load data of different format into numpy array, return data is in xyz

        f: the complete path to the file that you want to load

        """
        filename = osp.basename(f).lower()
        images = []

        # validate nii.gz on lung and mri with correct spacing_resample
        if filename.endswith((".nii", ".nii.gz", ".dcm")):
            if "radiopaedia" in filename or "corona" in filename:
                 f_nps = [nib.load(f).get_fdata(dtype=np.float32)]
            else:
                itkimage = sitk.ReadImage(f)
                if itkimage.GetDimension() == 4:
                    # slicer = sitk.ExtractImageFilter()
                    # s = list(itkimage.GetSize())
                    # s[-1] = 0
                    # slicer.SetSize(s)
                    # for slice_idx in range(itkimage.GetSize()[-1]):
                    #     slicer.SetIndex([0, 0, 0, slice_idx])
                    #     sitk_volume = slicer.Execute(itkimage)
                    #     images.append(sitk_volume)
                    images = [itkimage]
                else:
                    images = [itkimage]
                # print(images)
                # images = [sitk.DICOMOrient(img, 'LPS') for img in images]
                f_nps = [sitk.GetArrayFromImage(img) for img in images]

                # print(f_nps)
        return f_nps

    def load_save(self):
        """
        preprocess files, transfer to the correct type, and save it to the directory.
        """
        print(
            "Start convert images to numpy array using {}, please wait patiently"
            .format(self.gpu_tag))

        tic = time.time()
        # with open(self.dataset_json_path, 'r', encoding='utf-8') as f:
        #     dataset_json_dict = json.load(f)


        process_files = (self.image_files, self.label_files)
        process_tuple = ("images", "labels")
        save_tuple = (self.image_path, self.label_path)

        for i, files in enumerate(process_files):
            pre = self.preprocess[process_tuple[i]]
            savepath = save_tuple[i]

            for f in tqdm(
                    files,
                    total=len(files),
                    desc="preprocessing the {}".format(
                        ["images", "labels", "images_test"][i])):

                # load data will transpose the image from "zyx" to "xyz"
                spacing = (1,1,1)
                f_nps = Prep.load_medical_data(f)

                for volume_idx, f_np in enumerate(f_nps):
                    for op in pre:
                        if op.__name__ == "resample":
                            f_np, new_spacing = op(
                                f_np,
                                spacing=spacing)  # (960, 15, 960) if transpose
                        else:
                            f_np = op(f_np)

                    f_np = f_np.astype("float32") if i == 0 else f_np.astype(
                        "int32")
                    volume_idx = "" if len(f_nps) == 1 else f"-{volume_idx}"
                    np.save(
                        os.path.join(
                            savepath,
                            osp.basename(f).split(".")[0] + volume_idx), f_np)


        # with open(self.dataset_json_path, 'w', encoding='utf-8') as f:
        #     json.dump(dataset_json_dict, f, ensure_ascii=False, indent=4)

        print("The preprocess time on {} is {}".format(self.gpu_tag,
                                                       time.time() - tic))
    # TODO add data visualize method, such that data can be checked every time after preprocess.
    def visualize(self):
        pass
        # imga = Image.fromarray(np.int8(imga))
        # #当要保存的图片为灰度图像时，灰度图像的 numpy 尺度是 [1, h, w]。需要将 [1, h, w] 改变为 [h, w]
        # imgb = np.squeeze(imgb)

        # # imgb = Image.fromarray(np.int8(imgb))
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1,2,1),plt.xticks([]),plt.yticks([]),plt.imshow(imga)
        # plt.subplot(1,2,2),plt.xticks([]),plt.yticks([]),plt.imshow(imgb)
        # plt.show()

    @staticmethod
    def write_txt(txt, image_names, label_names=None):
        """
        write the image_names and label_names on the txt file like this:

        images/image_name labels/label_name
        ...

        or this when label is None.

        images/image_name
        ...

        """
        with open(txt, 'w') as f:
            for i in range(len(image_names)):
                if label_names is not None:
                    string = "{} {}\n".format('images/' + image_names[i],
                                              'labels/' + label_names[i])
                else:
                    string = "{}\n".format('images/' + image_names[i])

                f.write(string)

        print("successfully write to {}".format(txt))



class Prep_msd(Prep):
    def __init__(self, task_id):
        task_name = list(tasks[task_id].keys())[0].split('.')[0]
        print(f"Preparing task {task_id} {task_name}")
        super().__init__(
            dataset_root=f"PaddleSeg/contrib/MedicalSeg/data/{task_name}",
            raw_dataset_dir=f"{task_name}_raw/",
            images_dir=f"{task_name}/{task_name}/imagesTr",
            labels_dir=f"{task_name}/{task_name}/labelsTr",
            phase_dir=f"{task_name}_phase0/",
            urls=tasks[task_id],
            valid_suffix=("nii.gz", "nii.gz"),
            filter_key=(None, None),
            uncompress_params={"format": "tar",
                               "num_files": 1})

        self.preprocess = {
            "images": [
                # zscore
            ],
            "labels": [
            ]
        }

    def generate_txt(self, train_split=0.8,test_split=0.95):
        """generate the train_list.txt and val_list.txt"""

        txtname = [
            osp.join(self.phase_path, 'train_list.txt'),
            osp.join(self.phase_path, 'val_list.txt'),
            osp.join(self.phase_path, 'test_list.txt')
        ]

        image_files_npy = os.listdir(self.image_path)
        label_files_npy = os.listdir(self.label_path)

        self.split_files_txt(txtname[0], image_files_npy, label_files_npy,
                             train_split,test_split)
        self.split_files_txt(txtname[1], image_files_npy, label_files_npy,
                             train_split,test_split)
        self.split_files_txt(txtname[2], image_files_npy, label_files_npy,
                             train_split,test_split)


    def split_files_txt(self, txt, image_files, label_files=None, split=None,testsplit=None):
        print(22222222222222222222222222222222222)
        print(len(image_files))
        print(len(label_files))
        split = int(split * len(image_files))
        testsplit = int(testsplit * len(image_files))


        if "train" in txt:
            image_names = image_files[:split]
            label_names = label_files[:split]
        elif "val" in txt:
            # set the valset to 20% of images if all files need to be used in training
            image_names = image_files[split:testsplit]
            label_names = label_files[split:testsplit]
        elif "test" in txt:
            image_names = image_files[testsplit:]
            label_names = label_files[testsplit:]
        else:
            raise NotImplementedError(
                "The txt split except for train.txt, val.txt and test.txt is not implemented yet."
            )

        self.write_txt(txt, image_names, label_names)
                         


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Please provide task id. Example usage: \n\t python tools/prepare_msd.py 1 # for preparing MSD task 1"
        )

    try:
        task_id = int(sys.argv[1])
    except ValueError:
        print(
            f"Expecting number as command line argument, got {sys.argv[1]}.  Example usage: \n\t python tools/prepare_msd.py 1 # for preparing MSD task 1"
        )

    prep = Prep_msd(task_id)

    # json_path = osp.join(osp.dirname("data/Task01_BrainTumour/Task01_BrainTumour_raw/Task01_BrainTumour/Task01_BrainTumour/"), "dataset.json")
    # prep.generate_dataset_json(**parse_msd_basic_info(json_path))

    prep.load_save()
    prep.generate_txt()
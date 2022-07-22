# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import glob
import numpy as np
from paddleseg.datasets import Dataset
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose,MedicalCompose
from PIL import Image
import paddle

@manager.DATASETS.add_component
class CalDataset(paddle.io.Dataset):
    """

    
    Before use it . please run the cal_data_prepare.py in tools.

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): Cityscapes dataset directory.
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
    """

    def getleft(self):
        self.left_points = []
        for h_index in range(0, self.height, self.predict_size):
            if h_index + self.predict_size > self.height:
                h_index = self.height - self.predict_size
            for w_index in range(0, self.width, self.predict_size):
                if w_index + self.predict_size > self.width:
                    w_index = self.width - self.predict_size
                self.left_points.append([h_index, w_index])
        return self.left_points

    def getpaddings(self):
        self.padlist=[]
        padding_left = 0
        padding_right = 0
        padding_top = 0
        padding_bottom = 0
        for left_point in self.left_points:
            # 是否top方向进行镜像padding
            if (left_point[0] + self.predict_size + self.padding_size) > self.height:
                padding_bottom = 1
            # 是否right方向进行镜像padding
            if (left_point[1] + self.predict_size + self.padding_size) > self.width:
                padding_right = 1
            # 是否bottom方向进行镜像padding
            if (left_point[0] - self.predict_size) < 0:
                padding_top = 1
            # 是否left方向进行镜像padding
            if (left_point[1] - self.predict_size) < 0:
                padding_left = 1
            self.padlist.append([self.padding_size*padding_left,
                                 self.padding_size*padding_right,
                                 padding_top*self.padding_size,
                                 padding_bottom*self.padding_size])
    def create_lists(self):
        self.info_list=[]
        self.getleft()
        self.getpaddings()
        assert len(self.left_points)==len(self.padlist)
        for file in self.file_infos:
            for index in range(len(self.left_points)):
                data_dict={}
                data_dict['input_img']=file.split()[0]
                data_dict['label_img']=file.split()[1]
                data_dict['padding']=self.padlist[index]
                data_dict['left_point']=self.left_points[index]
                self.info_list.append(data_dict)



    def __init__(self, transforms, dataset_root, mode='train'):
        self.ignore_index = 255
        self.predict_size=388
        self.patch_size=512
        self.padding_size=92
        self.dataset_root = dataset_root
        self.transforms = MedicalCompose(transforms)
        self.file_list = list()
        mode = mode.lower()
        self.mode = mode

        if mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")
        if mode=="train":
            with open(os.path.join(dataset_root,"train.txt")) as f:
                self.file_infos=f.readlines()
        if mode=="val":
            with open(os.path.join(dataset_root,"val.txt")) as f:
                self.file_infos=f.readlines()
        self.height,self.width=np.load(os.path.join(dataset_root,self.file_infos[0].split()[0])).shape
        self.create_lists()

    def __getitem__(self, item):
        data_dict = self.info_list[item]
        input_array=np.load(os.path.join(self.dataset_root,data_dict['input_img'])).astype('float32')
        mask = np.asarray(Image.open(os.path.join(self.dataset_root,data_dict['label_img'])))
        left_h,left_w=data_dict["left_point"]
        padding_list=data_dict['padding']
        padding_left,padding_right,padding_top,padding_bottom=padding_list
        top=left_h if padding_top!=0 else (left_h-padding_top)
        bottom=left_h+self.predict_size if padding_bottom!=0 else (left_h+self.predict_size+self.padding_size)
        left=left_w if padding_left!=0 else (left_w-padding_left)
        right=left_w+self.predict_size if padding_right!=0 else (left_w+self.predict_size+self.padding_size)
        im=input_array[top:bottom,left:right]
        im=np.pad(im,((padding_top,padding_bottom),(padding_left,padding_right)),mode="reflect")

        if self.mode == 'test':
            im, _ = self.transforms(im=im)
            im = im[np.newaxis, ...]
            return im
        else:
            label=mask[left_h:(left_h+self.predict_size),left_w:(left_w+self.predict_size)]
            im, label = self.transforms(im=im, label=label)
            im = im[np.newaxis, ...]
            # if len(np.unique(label))>1:
            #     print("==================")
            return im, label

    def __len__(self):
        return len(self.info_list)

if __name__=="__main__":
    from paddleseg.transforms import transforms
    mytransform=[]
    testdataset=CalDataset(mytransform,dataset_root=r"D:\work\work001\Dataset\data")
    for index in range(len(testdataset)):
        print("imgshape:{},labelshape:{}".format(testdataset[index][0].shape,testdataset[index][1].shape))




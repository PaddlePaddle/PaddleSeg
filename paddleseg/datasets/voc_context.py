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

import tqdm
import numpy as np
from PIL import Image
from detail import Detail
from paddleseg.datasets import Dataset
from paddleseg.utils.download import download_file_and_uncompress,  _download_file
from paddleseg.utils import seg_env
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose

URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar"
JSON_URL = 'https://codalabuser.blob.core.windows.net/public/trainval_merged.json'

@manager.DATASETS.add_component
class PascalVOCContext(Dataset):
    """
    PascalVOC2010 dataset `http://host.robots.ox.ac.uk/pascal/VOC/`.
    If you want to use pascal context dataset, please run the voc_context.py in tools.

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): The dataset directory. Default: None
        mode (str): Which part of dataset to use. it is one of ('train', 'trainval', 'context', 'val').
            If you want to set mode to 'context', please make sure the dataset have been augmented. Default: 'train'.
    """

    def __init__(self, transforms=None, dataset_root=None, mode='train'):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        mode = mode.lower()
        self.mode = mode
        self.file_list = list()
        self.num_classes = 21
        self.ignore_index = 255

        if mode not in ['train', 'trainval', 'val']:
            raise ValueError(
                "`mode` should be one of ('train', 'trainval', 'val') in PascalVOCContext dataset, but got {}."
                .format(mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        if self.dataset_root is None:
            self.dataset_root = download_file_and_uncompress(
                url=URL,
                savepath=seg_env.DATA_HOME,
                extrapath=seg_env.DATA_HOME,
                extraname='VOCdevkit')
        elif not os.path.exists(self.dataset_root):
            self.dataset_root = os.path.normpath(self.dataset_root)
            savepath, extraname = self.dataset_root.rsplit(
                sep=os.path.sep, maxsplit=1)
            self.dataset_root = download_file_and_uncompress(
                url=URL,
                savepath=savepath,
                extrapath=savepath,
                extraname=extraname)
            
        #转换成context
        context_set_dir =  os.path.join(self.dataset_root, 'VOC2010')
        context_path = os.path.join(context_set_dir, 'context')
        if not os.path.exists(context_path):
            
            pascalcontext_seg_generator = PascalContextGenerator(context_set_dir, context_set_dir)
            pascalcontext_seg_generator.generate_label()

        image_set_dir = os.path.join(self.dataset_root, 'VOC2010', 'ImageSets','Segmentation')

        if mode == 'train':
            file_path = os.path.join(image_set_dir, 'train_context.txt')
        elif mode == 'val':
            file_path = os.path.join(image_set_dir, 'val_context.txt')
        elif mode == 'trainval':
            file_path = os.path.join(image_set_dir, 'trainval_context.txt')
       

        img_dir = os.path.join(self.dataset_root, 'VOC2010', 'JPEGImages')
        label_dir = os.path.join(self.dataset_root, 'VOC2010', 'context')

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                image_path = os.path.join(img_dir, ''.join([line, '.jpg']))
                label_path = os.path.join(label_dir, ''.join([line, '.png']))
                self.file_list.append([image_path, label_path])

                
class PascalContextGenerator(object):
    def __init__(self, voc_path, annotation_path):
        self.voc_path = voc_path
        self.annotation_path = annotation_path
        self.label_dir = os.path.join(self.voc_path, 'context')
        self._image_dir = os.path.join(self.voc_path, 'JPEGImages')
        self.annFile = os.path.join(self.annotation_path, 'trainval_merged.json')
        
        if not os.path.exists(self.annFile):
             _download_file(url=JSON_URL, savepath=self.annotation_path, print_progress=True)
            
        self._mapping = np.sort(np.array([
            0, 2, 259, 260, 415, 324, 9, 258, 144, 18, 19, 22,
            23, 397, 25, 284, 158, 159, 416, 33, 162, 420, 454, 295, 296,
            427, 44, 45, 46, 308, 59, 440, 445, 31, 232, 65, 354, 424,
            68, 326, 72, 458, 34, 207, 80, 355, 85, 347, 220, 349, 360,
            98, 187, 104, 105, 366, 189, 368, 113, 115]))
        self._key = np.array(range(len(self._mapping))).astype('uint8')

        self.train_detail = Detail(self.annFile, self._image_dir, 'train')
        self.train_ids = self.train_detail.getImgs()
        self.val_detail = Detail(self.annFile, self._image_dir, 'val')
        self.val_ids = self.val_detail.getImgs()

        if not os.path.exists(self.label_dir):
            os.makedirs(self.label_dir)

    def _class_to_index(self, mask, _mapping, _key):
        # assert the values
        values = np.unique(mask)
        for i in range(len(values)):
            assert (values[i] in _mapping)
        index = np.digitize(mask.ravel(), _mapping, right=True)
        return _key[index].reshape(mask.shape)
    
    def save_mask(self, img_id, mode):
        if mode == 'train':
            mask = Image.fromarray(self._class_to_index(self.train_detail.getMask(img_id), _mapping=self._mapping, _key=self._key))
        elif mode == 'val':
            mask = Image.fromarray(self._class_to_index(self.val_detail.getMask(img_id), _mapping=self._mapping, _key=self._key))
        filename = img_id['file_name']
        basename, _ = os.path.splitext(filename)
        if filename.endswith(".jpg"):
            mask_png_name = basename + '.png'
            mask.save(os.path.join(self.label_dir, mask_png_name))
        return basename

    def generate_label(self):
        with open(os.path.join(self.voc_path, 'ImageSets/Segmentation/train_context.txt'), 'w') as f:
            for img_id in tqdm.tqdm(self.train_ids, desc='train'):
                basename = self.save_mask(img_id, 'train')
                f.writelines(''.join([basename, '\n']))

        with open(os.path.join(self.voc_path, 'ImageSets/Segmentation/val_context.txt'), 'w') as f:
            for img_id in tqdm.tqdm(self.val_ids, desc='val'):
                basename = self.save_mask(img_id, 'val')
                f.writelines(''.join([basename, '\n']))
                
        with open(os.path.join(self.voc_path, 'ImageSets/Segmentation/trainval_context.txt'), 'w') as f:
            for img in tqdm.tqdm(os.listdir(self.label_dir), desc='trainval'):
                if img.endswith('.png'):
                    basename = img.split('.', 1)[0]
                    f.writelines(''.join([basename, '\n']))
                    

if __name__ == '__main__':
    PascalVOCContext()

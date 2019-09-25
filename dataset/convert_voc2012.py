# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import sys
import os
import numpy as np
import os
from PIL import Image
import glob

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
def remove_colormap(filename):
    gray_anno = np.array(Image.open(filename))
    return gray_anno


def save_annotation(annotation, filename):
    annotation = annotation.astype(dtype=np.uint8)
    annotation = Image.fromarray(annotation)
    annotation.save(filename)

def convert_list(origin_file, seg_file, output_folder):
    with open(seg_file, 'w') as fid_seg:
        with open(origin_file) as fid_ori:
            lines = fid_ori.readlines()
            for line in lines:
                line = line.strip()
                line = '.'.join([line, 'jpg'])
                img_name = os.path.join("JPEGImages", line)
                line = line.replace('jpg', 'png')
                anno_name = os.path.join(output_folder.split(os.sep)[-1], line)
                new_line = ' '.join([img_name, anno_name])
                fid_seg.write(new_line + "\n")

if __name__ == "__main__":
    pascal_root = "./VOCtrainval_11-May-2012/VOC2012"
    pascal_root = os.path.join(LOCAL_PATH, pascal_root)
    seg_folder = os.path.join(pascal_root, "SegmentationClass")
    txt_folder = os.path.join(pascal_root, "ImageSets/Segmentation")
    train_path = os.path.join(txt_folder, "train.txt")
    val_path = os.path.join(txt_folder, "val.txt")
    trainval_path = os.path.join(txt_folder, "trainval.txt")

    # 标注图转换后存储目录
    output_folder = os.path.join(pascal_root, "SegmentationClassAug")
    
    print("annotation convert and file list convert")
    if not os.path.exists(os.path.join(LOCAL_PATH, output_folder)):
        os.mkdir(os.path.join(LOCAL_PATH, output_folder))
    annotation_names = glob.glob(os.path.join(seg_folder, '*.png'))
    for annotation_name in annotation_names:
        annotation = remove_colormap(annotation_name)
        filename = os.path.basename(annotation_name)
        save_name = os.path.join(output_folder, filename)
        save_annotation(annotation, save_name)

    convert_list(train_path, train_path.replace('txt', 'list'), output_folder)
    convert_list(val_path, val_path.replace('txt', 'list'), output_folder)
    convert_list(trainval_path, trainval_path.replace('txt', 'list'), output_folder)


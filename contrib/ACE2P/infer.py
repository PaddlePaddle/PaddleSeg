# coding: utf8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import numpy as np
from utils.util import get_arguments
from utils.palette import get_palette
from PIL import Image as PILImage
import importlib

args = get_arguments()
config = importlib.import_module('config')
cfg = getattr(config, 'cfg')

# paddle垃圾回收策略FLAG，ACE2P模型较大，当显存不够时建议开启
os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'

import paddle.fluid as fluid


# 预测数据集类
class TestDataSet():
    def __init__(self):
        self.data_dir = cfg.data_dir
        self.data_list_file = cfg.data_list_file
        self.data_list = self.get_data_list()
        self.data_num = len(self.data_list)

    def get_data_list(self):
        # 获取预测图像路径列表
        data_list = []
        data_file_handler = open(self.data_list_file, 'r')
        for line in data_file_handler:
            img_name = line.strip()
            name_prefix = img_name.split('.')[0]
            if len(img_name.split('.')) == 1:
                img_name = img_name + '.jpg'
            img_path = os.path.join(self.data_dir, img_name)
            data_list.append(img_path)
        return data_list

    def preprocess(self, img):
        # 图像预处理
        if cfg.example == 'ACE2P':
            reader = importlib.import_module('reader')
            ACE2P_preprocess = getattr(reader, 'preprocess')
            img = ACE2P_preprocess(img)
        else:
            img = cv2.resize(img, cfg.input_size).astype(np.float32)
            img -= np.array(cfg.MEAN)
            img /= np.array(cfg.STD)
            img = img.transpose((2, 0, 1))
            img = np.expand_dims(img, axis=0)
        return img

    def get_data(self, index):
        # 获取图像信息
        img_path = self.data_list[index]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            return img, img, img_path, None

        img_name = img_path.split(os.sep)[-1]
        name_prefix = img_name.replace('.' + img_name.split('.')[-1], '')
        img_shape = img.shape[:2]
        img_process = self.preprocess(img)

        return img, img_process, name_prefix, img_shape


def infer():
    if not os.path.exists(cfg.vis_dir):
        os.makedirs(cfg.vis_dir)
    palette = get_palette(cfg.class_num)
    # 人像分割结果显示阈值
    thresh = 120

    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # 加载预测模型
    test_prog, feed_name, fetch_list = fluid.io.load_inference_model(
        dirname=cfg.model_path, executor=exe, params_filename='__params__')

    #加载预测数据集
    test_dataset = TestDataSet()
    data_num = test_dataset.data_num

    for idx in range(data_num):
        # 数据获取
        ori_img, image, im_name, im_shape = test_dataset.get_data(idx)
        if image is None:
            print(im_name, 'is None')
            continue

        # 预测
        if cfg.example == 'ACE2P':
            # ACE2P模型使用多尺度预测
            reader = importlib.import_module('reader')
            multi_scale_test = getattr(reader, 'multi_scale_test')
            parsing, logits = multi_scale_test(exe, test_prog, feed_name,
                                               fetch_list, image, im_shape)
        else:
            # HumanSeg,RoadLine模型单尺度预测
            result = exe.run(
                program=test_prog,
                feed={feed_name[0]: image},
                fetch_list=fetch_list)
            parsing = np.argmax(result[0][0], axis=0)
            parsing = cv2.resize(parsing.astype(np.uint8), im_shape[::-1])

        # 预测结果保存
        result_path = os.path.join(cfg.vis_dir, im_name + '.png')
        if cfg.example == 'HumanSeg':
            logits = result[0][0][1] * 255
            logits = cv2.resize(logits, im_shape[::-1])
            ret, logits = cv2.threshold(logits, thresh, 0, cv2.THRESH_TOZERO)
            logits = 255 * (logits - thresh) / (255 - thresh)
            # 将分割结果添加到alpha通道
            rgba = np.concatenate((ori_img, np.expand_dims(logits, axis=2)),
                                  axis=2)
            cv2.imwrite(result_path, rgba)
        else:
            output_im = PILImage.fromarray(np.asarray(parsing, dtype=np.uint8))
            output_im.putpalette(palette)
            output_im.save(result_path)

        if (idx + 1) % 100 == 0:
            print('%d  processd' % (idx + 1))

    print('%d  processd done' % (idx + 1))

    return 0


if __name__ == "__main__":
    infer()

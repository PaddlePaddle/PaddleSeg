# -*- coding: utf-8 -*-
import os
import numpy as np
from utils.util import get_arguments
from utils.palette import get_palette
from utils.data_util import Cluster, pad_img
from PIL import Image as PILImage
import importlib
import paddle.fluid as fluid

args = get_arguments()
config = importlib.import_module('config')
cfg = getattr(config, 'cfg')

cluster = Cluster()

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
        h, w = img.shape[:2]
        h_new = (h//32 + 1 if h % 32 != 0 else h//32)*32
        w_new = (w//32 + 1 if w % 32 != 0 else w//32)*32
        img = np.pad(img, ((0, h_new - h), (0, w_new - w), (0, 0)), 'edge')
        
        img = img.astype(np.float32)/255.0
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def get_data(self, index):
        # 获取图像信息
        img_path = self.data_list[index]
        img = np.array(PILImage.open(img_path))
        if img is None:
            return img, img,img_path, None

        img_name = img_path.split(os.sep)[-1]
        name_prefix = img_name.replace('.'+img_name.split('.')[-1],'')
        img_shape = img.shape[:2]
        img_process = self.preprocess(img)

        return img_process, name_prefix, img_shape


def infer():
    if not os.path.exists(cfg.vis_dir):
        os.makedirs(cfg.vis_dir)

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
        image, im_name, im_shape = test_dataset.get_data(idx)
        if image is None:
            print(im_name, 'is None')
            continue
        # 预测
        output = exe.run(program=test_prog, feed={feed_name[0]: image}, fetch_list=fetch_list)
        instance_map, predictions = cluster.cluster(output[0][0], n_sigma=cfg.n_sigma, \
                                    min_pixel=cfg.min_pixel, threshold=cfg.threshold)

        # 预测结果保存
        instance_map = pad_img(instance_map, image.shape[2:])
        instance_map = instance_map[:im_shape[0], :im_shape[1]]
        output_im = PILImage.fromarray(np.asarray(instance_map, dtype=np.uint8))
        palette = get_palette(len(predictions) + 1)
        output_im.putpalette(palette)
        result_path = os.path.join(cfg.vis_dir, im_name+'.png')
        output_im.save(result_path)

        if (idx + 1) % 100 == 0:
            print('%d  processd' % (idx + 1))
            
    print('%d  processd done' % (idx + 1))   
    
    return 0


if __name__ == "__main__":
    infer()

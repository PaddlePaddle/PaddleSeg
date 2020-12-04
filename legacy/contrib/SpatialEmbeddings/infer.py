# -*- coding: utf-8 -*-
import os
import numpy as np
from utils.util import get_arguments
from utils.palette import get_palette
from utils.data_util import Cluster, pad_img
from PIL import Image as PILImage
import importlib
import paddle.fluid as fluid
from models import SpatialEmbeddings

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
        h_new, w_new = cfg.input_size
        img = np.pad(img, ((0, h_new - h), (0, w_new - w), (0, 0)), 'edge')
        img = img.astype(np.float32) / 255.0
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def get_data(self, index):
        # 获取图像信息
        img_path = self.data_list[index]
        img = np.array(PILImage.open(img_path))
        if img is None:
            return img, img, img_path, None

        img_name = img_path.split(os.sep)[-1]
        name_prefix = img_name.replace('.' + img_name.split('.')[-1], '')
        img_shape = img.shape[:2]
        img_process = self.preprocess(img)

        return img_process, name_prefix, img_shape


def get_model(main_prog, startup_prog):
    img_shape = [3, cfg.input_size[0], cfg.input_size[1]]
    with fluid.program_guard(main_prog, startup_prog):
        with fluid.unique_name.guard():
            input = fluid.layers.data(
                name='image', shape=img_shape, dtype='float32')
            output = SpatialEmbeddings(input)
    return input, output


def infer():
    if not os.path.exists(cfg.vis_dir):
        os.makedirs(cfg.vis_dir)

    startup_prog = fluid.Program()
    test_prog = fluid.Program()

    input, output = get_model(test_prog, startup_prog)
    test_prog = test_prog.clone(for_test=True)

    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if not os.path.exists(cfg.model_path):
        raise RuntimeError('No pre-trained model found under path {}'.format(
            cfg.model_path))

    # 加载预测模型
    def if_exist(var):
        return os.path.exists(os.path.join(cfg.model_path, var.name))

    fluid.io.load_vars(
        exe, cfg.model_path, main_program=test_prog, predicate=if_exist)

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
        outputs = exe.run(
            program=test_prog, feed={'image': image}, fetch_list=output)
        instance_map, predictions = cluster.cluster(outputs[0][0], n_sigma=cfg.n_sigma, \
                                    min_pixel=cfg.min_pixel, threshold=cfg.threshold)

        # 预测结果保存
        instance_map = pad_img(instance_map, image.shape[2:])
        instance_map = instance_map[:im_shape[0], :im_shape[1]]
        output_im = PILImage.fromarray(np.asarray(instance_map, dtype=np.uint8))
        palette = get_palette(len(predictions) + 1)
        output_im.putpalette(palette)
        result_path = os.path.join(cfg.vis_dir, im_name + '.png')
        output_im.save(result_path)

        if (idx + 1) % 100 == 0:
            print('%d  processd' % (idx + 1))

    print('%d  processd done' % (idx + 1))

    return 0


if __name__ == "__main__":
    infer()

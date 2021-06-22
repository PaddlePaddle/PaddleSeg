# coding: utf-8
import os
import cv2
import requests
import json
import base64
import numpy as np
import time
import threading
import re

#分割服务的地址
IMAGE_SEG_URL = 'http://xxx.xxx.xxx.xxx:8010/ImageSegService/inference'


class ClientThread(threading.Thread):
    def __init__(self, thread_id, image_data_repo):
        threading.Thread.__init__(self)
        self.__thread_id = thread_id
        self.__image_data_repo = image_data_repo

    def run(self):
        self.__request_image_seg_service()

    def __request_image_seg_service(self):
        # 持续发送150个请求
        for i in range(1, 151):
            print("Epoch %d, thread %d" % (i, self.__thread_id))
            self.__benchmark_test()

    # benchmark test
    def __benchmark_test(self):
        start = time.time()
        for image_filename in self.__image_data_repo:
            mask_mat_list = self.__request_predictor_server(image_filename)
            input_img = self.__image_data_repo.get_image_matrix(image_filename)
            # 将获得的mask matrix转换成可视化图像，并在当前目录下保存为图像文件
            # 如果进行压测，可以把这句话注释掉
            for j in range(len(mask_mat_list)):
                self.__visualization(mask_mat_list[j], image_filename, 2,
                                     input_img)
        latency = time.time() - start
        print("total latency = %f s" % (latency))

    # 对预测结果进行可视化
    # input_raw_mask 是server返回的预测结果
    # output_img 是可视化结果存储路径
    def __visualization(self, mask_mat, output_img, num_cls, input_img):
        # ColorMap for visualization more clearly
        n = num_cls
        color_map = []
        for j in range(n):
            lab = j
            a = b = c = 0
            color_map.append([a, b, c])
            i = 0
            while lab:
                color_map[j][0] |= (((lab >> 0) & 1) << (7 - i))
                color_map[j][1] |= (((lab >> 1) & 1) << (7 - i))
                color_map[j][2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        im = cv2.imdecode(mask_mat, 1)
        w, h, c = im.shape
        im2 = cv2.resize(im, (w, h))
        im = im2
        # I = aF + (1-a)B
        a = im / 255.0
        im = a * input_img + (1 - a) * [255, 255, 255]
        cv2.imwrite(output_img, im)

    def __request_predictor_server(self, input_img):
        data = {"instances": [self.__get_item_json(input_img)]}
        response = requests.post(IMAGE_SEG_URL, data=json.dumps(data))
        try:
            response = json.loads(response.text)
            prediction_list = response["prediction"]
            mask_response_list = [
                mask_response["info"] for mask_response in prediction_list
            ]
            mask_raw_list = [
                json.loads(mask_response)["mask"]
                for mask_response in mask_response_list
            ]
        except Exception as err:
            print(
                "Exception[%s], server_message[%s]" % (str(err), response.text))
            return None
        # 使用 json 协议回复的包也是 base64 编码过的
        mask_binary_list = [
            base64.b64decode(mask_raw) for mask_raw in mask_raw_list
        ]
        m = [
            np.fromstring(mask_binary, np.uint8)
            for mask_binary in mask_binary_list
        ]
        return m

    # 请求预测服务
    # input_img 要预测的图片列表
    def __get_item_json(self, input_img):
        # 使用 http 协议请求服务时, 请使用 base64 编码发送图片
        item_binary_b64 = str(
            base64.b64encode(
                self.__image_data_repo.get_image_binary(input_img)), 'utf-8')
        item_size = len(item_binary_b64)
        item_json = {"image_length": item_size, "image_binary": item_binary_b64}
        return item_json


def create_thread_pool(thread_num, image_data_repo):
    return [ClientThread(i + 1, image_data_repo) for i in range(thread_num)]


def run_threads(thread_pool):
    for thread in thread_pool:
        thread.start()

    for thread in thread_pool:
        thread.join()


class ImageDataRepo:
    def __init__(self, dir_name):
        print("Loading images data...")
        self.__data = {}
        pattern = re.compile(".+\.(jpg|jpeg)", re.I)
        if os.path.isdir(dir_name):
            for image_filename in os.listdir(dir_name):
                if pattern.match(image_filename):
                    full_path = os.path.join(dir_name, image_filename)
                    fp = open(full_path, mode="rb")
                    image_binary_data = fp.read()
                    image_mat_data = cv2.imread(full_path)
                    self.__data[image_filename] = (image_binary_data,
                                                   image_mat_data)
        else:
            raise Exception("Please use directory to initialize")
        print("Finish loading.")

    def __iter__(self):
        for filename in self.__data:
            yield filename

    def get_image_binary(self, image_name):
        return self.__data[image_name][0]

    def get_image_matrix(self, image_name):
        return self.__data[image_name][1]


if __name__ == "__main__":
    #preprocess
    IDR = ImageDataRepo("images")
    thread_pool = create_thread_pool(thread_num=1, image_data_repo=IDR)
    run_threads(thread_pool)

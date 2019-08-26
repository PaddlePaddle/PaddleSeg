# coding: utf-8
import sys

import cv2
import requests
import json
import base64
import numpy as np
import time
import threading

#分割服务的地址
#IMAGE_SEG_URL = 'http://yq01-gpu-151-23-00.epc:8010/ImageSegService/inference'
#IMAGE_SEG_URL = 'http://106.12.25.202:8010/ImageSegService/inference'
IMAGE_SEG_URL = 'http://180.76.118.53:8010/ImageSegService/inference'

# 请求预测服务
# input_img 要预测的图片列表
def get_item_json(input_img):
    with open(input_img, mode="rb") as fp:
        # 使用 http 协议请求服务时, 请使用 base64 编码发送图片
        item_binary_b64 = str(base64.b64encode(fp.read()), 'utf-8')
        item_size = len(item_binary_b64)
        item_json = {
            "image_length": item_size,
            "image_binary": item_binary_b64
        }
        return item_json


def request_predictor_server(input_img_list, dir_name):
    data = {"instances" : [get_item_json(dir_name + input_img) for input_img in input_img_list]}
    response = requests.post(IMAGE_SEG_URL, data=json.dumps(data))
    try:
        response = json.loads(response.text)
        prediction_list = response["prediction"]
        mask_response_list = [mask_response["info"] for mask_response in prediction_list]
        mask_raw_list = [json.loads(mask_response)["mask"] for mask_response in mask_response_list]
    except Exception as err:
        print ("Exception[%s], server_message[%s]" % (str(err), response.text))
        return None
    # 使用 json 协议回复的包也是 base64 编码过的
    mask_binary_list = [base64.b64decode(mask_raw) for mask_raw in mask_raw_list]
    m = [np.fromstring(mask_binary, np.uint8) for mask_binary in mask_binary_list]
    return m

# 对预测结果进行可视化
# input_raw_mask 是server返回的预测结果
# output_img 是可视化结果存储路径
def visualization(mask_mat, output_img):
    # ColorMap for visualization more clearly
    color_map = [[128, 64, 128],
                 [244, 35, 231],
                 [69, 69, 69],
                 [102, 102, 156],
                 [190, 153, 153],
                 [153, 153, 153],
                 [250, 170, 29],
                 [219, 219, 0],
                 [106, 142, 35],
                 [152, 250, 152],
                 [69, 129, 180],
                 [219, 19, 60],
                 [255, 0, 0],
                 [0, 0, 142],
                 [0, 0, 69],
                 [0, 60, 100],
                 [0, 79, 100],
                 [0, 0, 230],
                 [119, 10, 32]]

    im = cv2.imdecode(mask_mat, 1)
    w, h, c = im.shape
    im2 = cv2.resize(im, (w, h))
    im = im2
    for i in range(0, h):
        for j in range(0, w):
            im[i, j] = color_map[im[i, j, 0]]
    cv2.imwrite(output_img, im)


#benchmark test
def benchmark_test(batch_size, img_list):
    start = time.time()
    total_size = len(img_list)
    for i in range(0, total_size, batch_size):
        mask_mat_list = request_predictor_server(img_list[i : np.min([i + batch_size, total_size])], "images/")
        # 将获得的mask matrix转换成可视化图像，并在当前目录下保存为图像文件
        # 如果进行压测，可以把这句话注释掉
        # for j in range(len(mask_mat_list)):
        #     visualization(mask_mat_list[j], img_list[j + i])
    latency = time.time() - start
    print("batch size = %d, total latency = %f s" % (batch_size, latency))


class ClientThread(threading.Thread):
    def __init__(self, thread_id, batch_size):
        threading.Thread.__init__(self)
        self.__thread_id = thread_id
        self.__batch_size = batch_size

    def run(self):
        self.request_image_seg_service(3)

    def request_image_seg_service(self, imgs_num):
        total_size = imgs_num
        img_list = [str(i + 1) + ".jpg" for i in range(total_size)]
        # batch_size_list = [2**i for i in range(0, 4)]
        # 持续发送150个请求
        batch_size_list = [self.__batch_size] * 150
        i = 1
        for batch_size in batch_size_list:
            print("Epoch %d, thread %d" % (i, self.__thread_id))
            i += 1
            benchmark_test(batch_size, img_list)


def create_thread_pool(thread_num, batch_size):
    return [ClientThread(i + 1, batch_size) for i in range(thread_num)]


def run_threads(thread_pool):
    for thread in thread_pool:
        thread.start()

    for thread in thread_pool:
        thread.join()

if __name__ == "__main__":
    thread_pool = create_thread_pool(thread_num=2, batch_size=1)
    run_threads(thread_pool)


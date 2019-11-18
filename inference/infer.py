# coding: utf8
# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
import sys
import time

import gflags
import yaml
import numpy as np
import cv2

from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import NativeConfig
from paddle.fluid.core import create_paddle_predictor
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import PaddleBuf
from paddle.fluid.core import PaddleDType
from concurrent.futures import ThreadPoolExecutor, as_completed

gflags.DEFINE_string("conf", default="", help="Configuration File Path")
gflags.DEFINE_string("input_dir", default="", help="Directory of Input Images")
gflags.DEFINE_boolean("use_pr", default=False, help="Use optimized model")
Flags = gflags.FLAGS

# ColorMap for visualization more clearly
color_map = [[128, 64, 128], [244, 35, 231], [69, 69, 69], [102, 102, 156],
             [190, 153, 153], [153, 153, 153], [250, 170, 29], [219, 219, 0],
             [106, 142, 35], [152, 250, 152], [69, 129, 180], [219, 19, 60],
             [255, 0, 0], [0, 0, 142], [0, 0, 69], [0, 60, 100], [0, 79, 100],
             [0, 0, 230], [119, 10, 32]]


class ConfItemsNotFoundError(Exception):
    def __init__(self, message):
        super().__init__(message + " item not Found")


class Config:
    def __init__(self, config_dict):
        if "DEPLOY" not in config_dict:
            raise ConfItemsNotFoundError("DEPLOY")
        deploy_dict = config_dict["DEPLOY"]
        if "EVAL_CROP_SIZE" not in deploy_dict:
            raise ConfItemsNotFoundError("EVAL_CROP_SIZE")
        # 1. get resize
        self.resize = [int(value) for value in
                       deploy_dict["EVAL_CROP_SIZE"].strip("()").split(",")]

        # 2. get mean
        if "MEAN" not in deploy_dict:
            raise ConfItemsNotFoundError("MEAN")
        self.mean = deploy_dict["MEAN"]

        # 3. get std
        if "STD" not in deploy_dict:
            raise ConfItemsNotFoundError("STD")
        self.std = deploy_dict["STD"]

        # 4. get image type
        if "IMAGE_TYPE" not in deploy_dict:
            raise ConfItemsNotFoundError("IMAGE_TYPE")
        self.img_type = deploy_dict["IMAGE_TYPE"]

        # 5. get class number
        if "NUM_CLASSES" not in deploy_dict:
            raise ConfItemsNotFoundError("NUM_CLASSES")
        self.class_num = deploy_dict["NUM_CLASSES"]

        # 7. set model path
        if "MODEL_PATH" not in deploy_dict:
            raise ConfItemsNotFoundError("MODEL_PATH")
        self.model_path = deploy_dict["MODEL_PATH"]

        # 8. get model file_name
        if "MODEL_FILENAME" not in deploy_dict:
            self.model_file_name = "__model__"
        else:
            self.model_file_name = deploy_dict["MODEL_FILENAME"]

        # 9. get model param file name
        if "PARAMS_FILENAME" not in deploy_dict:
            self.param_file_name = "__params__"
        else:
            self.param_file_name = deploy_dict["PARAMS_FILENAME"]

        # 10. get pre_processor
        if "PRE_PROCESSOR" not in deploy_dict:
            raise ConfItemsNotFoundError("PRE_PROCESSOR")
        self.pre_processor = deploy_dict["PRE_PROCESSOR"]

        # 11. use_gpu
        if "USE_GPU" not in deploy_dict:
            self.use_gpu = 0
        else:
            self.use_gpu = deploy_dict["USE_GPU"]

        # 12. predictor_mode
        if "PREDICTOR_MODE" not in deploy_dict:
            raise ConfItemsNotFoundError("PREDICTOR_MODE")
        self.predictor_mode = deploy_dict["PREDICTOR_MODE"]

        # 13. batch_size
        if "BATCH_SIZE" not in deploy_dict:
            raise ConfItemsNotFoundError("BATCH_SIZE")
        self.batch_size = deploy_dict["BATCH_SIZE"]

        # 14. channels
        if "CHANNELS" not in deploy_dict:
            raise ConfItemsNotFoundError("CHANNELS")
        self.channels = deploy_dict["CHANNELS"]


class PreProcessor:
    def __init__(self, config):
        self.resize_size = (config.resize[0], config.resize[1])
        self.mean = config.mean
        self.std = config.std

    def process(self, image_file, im_list, ori_h_list, ori_w_list, idx, use_pr=False):
        start = time.time()
        im = cv2.imread(image_file, -1)
        end = time.time()
        print("imread spent %fs" % (end - start))
        channels = im.shape[2]
        ori_h = im.shape[0]
        ori_w = im.shape[1]
        if channels == 1:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            channels = im.shape[2]
        if channels != 3 and channels != 4:
            print("Only support rgb(gray) or rgba image.")
            return -1

        if ori_h != self.resize_size[0] or ori_w != self.resize_size[1]:
            start = time.time()
            im = cv2.resize(im, self.resize_size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
            end = time.time()
            print("resize spent %fs" % (end - start))
        if not use_pr:
            start = time.time()
            im_mean = np.array(self.mean).reshape((3, 1, 1))
            im_std = np.array(self.std).reshape((3, 1, 1))

            # HWC -> CHW, don't use transpose((2, 0, 1))
            im = im.swapaxes(1, 2)
            im = im.swapaxes(0, 1)
            im = im[:, :, :].astype('float32') / 255.0
            im -= im_mean
            im /= im_std
            end = time.time()
            print("preprocessing spent %fs" % (end-start))
        im = im[np.newaxis,:,:,:]
        im_list[idx] = im
        ori_h_list[idx] = ori_h
        ori_w_list[idx] = ori_w
        return im, ori_h, ori_w


class Predictor:
    def __init__(self, config):
        self.config = config
        model_file = os.path.join(config.model_path, config.model_file_name)
        param_file = os.path.join(config.model_path, config.param_file_name)
        if config.predictor_mode == "NATIVE":
            predictor_config = NativeConfig()
            predictor_config.prog_file = model_file
            predictor_config.param_file = param_file
            predictor_config.use_gpu = config.use_gpu
            predictor_config.device = 0
            predictor_config.fraction_of_gpu_memory = 0
        elif config.predictor_mode == "ANALYSIS":
            predictor_config = AnalysisConfig(model_file, param_file)
            if config.use_gpu:
                predictor_config.enable_use_gpu(100, 0)
            else:
                predictor_config.disable_gpu()
            # need to use zero copy run
            # predictor_config.switch_use_feed_fetch_ops(False)
            # predictor_config.enable_tensorrt_engine(
            #     workspace_size=1<<30,
            #     max_batch_size=1,
            #     min_subgraph_size=3,
            #     precision_mode=AnalysisConfig.Precision.Int8,
            #     use_static=False,
            #     use_calib_mode=True
            # )
            predictor_config.switch_specify_input_names(True)
            predictor_config.enable_memory_optim()

        self.predictor = create_paddle_predictor(predictor_config)
        self.preprocessor = PreProcessor(config)
        self.threads_pool = ThreadPoolExecutor(config.batch_size)

    def make_tensor(self, inputs, batch_size, use_pr=False):
        im_tensor = PaddleTensor()
        im_tensor.name = "image"
        if not use_pr:
            im_tensor.shape = [batch_size, self.config.channels,
                               self.config.resize[1], self.config.resize[0]]
        else:
            im_tensor.shape = [batch_size, self.config.resize[1],
                               self.config.resize[0], self.config.channels]
        print(im_tensor.shape)
        im_tensor.dtype = PaddleDType.FLOAT32
        start = time.time()
        im_tensor.data = PaddleBuf(inputs.ravel().astype("float32"))
        print("flatten time: %f" % (time.time() - start))
        return [im_tensor]

    def output_result(self, image_name, output, ori_h, ori_w, use_pr=False):
        mask = output
        if not use_pr:
            mask = np.argmax(output, axis=0)
        mask = mask.astype('uint8')
        mask_png = mask
        score_png = mask_png[:, :, np.newaxis]
        score_png = np.concatenate([score_png] * 3, axis=2)
        for i in range(score_png.shape[0]):
            for j in range(score_png.shape[1]):
                score_png[i, j] = color_map[score_png[i, j, 0]]

        mask_save_name = image_name + ".png"
        cv2.imwrite(mask_save_name, mask_png, [cv2.CV_8UC1])

        result_name = image_name + "_result.png"
        result_png = score_png
        # if not use_pr:
        result_png = cv2.resize(result_png, (ori_w, ori_h), fx=0, fy=0,
                                interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(result_name, result_png, [cv2.CV_8UC1])

        print("save result of [" + image_name + "] done.")

    def predict(self, images):
        batch_size = self.config.batch_size
        total_runtime = 0
        total_imwrite_time = 0
        for i in range(0, len(images), batch_size):
            start = time.time()
            bz = batch_size
            if i + batch_size >= len(images):
                bz = len(images) - i
            im_list = [0] * bz
            ori_h_list = [0] * bz
            ori_w_list = [0] * bz
            tasks = [self.threads_pool.submit(self.preprocessor.process,
                                              images[i + j], im_list,
                                              ori_h_list, ori_w_list,
                                              j, Flags.use_pr)
                     for j in range(bz)]
            # join all running threads
            for t in as_completed(tasks):
                pass
            input_data = np.concatenate(im_list)
            input_data = self.make_tensor(input_data, bz, use_pr=Flags.use_pr)
            inference_start = time.time()
            output_data = self.predictor.run(input_data)[0]
            end = time.time()
            print("inference time = %fs " % (end - inference_start))
            print("runtime = %fs " % (end - start))
            total_runtime += (end - start)
            output_data = output_data.as_ndarray()

            output_start = time.time()
            for j in range(bz):
                self.output_result(images[i + j], output_data[j],
                                   ori_h_list[j], ori_w_list[j], Flags.use_pr)
            output_end = time.time()
            total_imwrite_time += output_end - output_start
        print("total time = %fs" % total_runtime)
        print("total imwrite time = %fs" % total_imwrite_time)


def usage():
    print("Usage: python infer.py --conf=/config/path/to/your/model " +
          "--input_dir=/directory/of/your/input/images [--use_pr=True]")


def read_conf(conf_file):
    if not os.path.exists(conf_file):
        raise FileNotFoundError("Can't find the configuration file path," +
                                " please check whether the configuration" +
                                " path is correctly set.")
    f = open(conf_file)
    config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = Config(config_dict)
    return config


def read_input_dir(input_dir, ext=".jpg|.jpeg"):
    if not os.path.exists(input_dir):
        raise FileNotFoundError("This input directory doesn't exist, please" +
                                " check whether the input directory is" +
                                " correctly set.")
    if not os.path.isdir(input_dir):
        raise NotADirectoryError("This input directory in not a directory," +
                                 " please check whether the input directory" +
                                 " is correctly set.")
    files_list = []
    ext_list = ext.split("|")
    files = os.listdir(input_dir)
    for file in files:
        for ext_suffix in ext_list:
            if file.endswith(ext_suffix):
                full_path = os.path.join(input_dir, file)
                files_list.append(full_path)
                break
    return files_list


def main(argv):
    # 0. parse the argument
    Flags(argv)
    if Flags.conf == "" or Flags.input_dir == "":
        usage()
        return -1
    try:
        # 1. get a conf dictionary
        seg_deploy_configs = read_conf(Flags.conf)
        # 2. get all the images path with extension '.jpeg' at input_dir
        images = read_input_dir(Flags.input_dir)
        if len(images) == 0:
            print("No Images Found! Please check whether the images format" +
                  " is correct. Supporting format: [.jpeg|.jpg].")
        print(images)
    except Exception as e:
        print(e)
        return -1

    # 3. init predictor and predict
    seg_predictor = Predictor(seg_deploy_configs)
    seg_predictor.predict(images)


if __name__ == "__main__":
    main(sys.argv)

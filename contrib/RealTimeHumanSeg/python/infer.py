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

import numpy as np
import cv2

import paddle.fluid as fluid


def load_model(model_dir, use_gpu=False):
    """
    Load model files and init paddle predictor
    """
    prog_file = os.path.join(model_dir, '__model__')
    params_file = os.path.join(model_dir, '__params__')
    config = fluid.core.AnalysisConfig(prog_file, params_file)
    if use_gpu:
        config.enable_use_gpu(100, 0)
        config.switch_ir_optim(True)
    else:
        config.disable_gpu()
    config.disable_glog_info()
    config.switch_specify_input_names(True)
    config.enable_memory_optim()
    return fluid.core.create_paddle_predictor(config)


class HumanSeg:
    """
    Human Segmentation Class
    """
    def __init__(self, model_dir, mean, scale, eval_size, use_gpu=False):
        self.mean = np.array(mean).reshape((3, 1, 1))
        self.scale = np.array(scale).reshape((3, 1, 1))
        self.eval_size = eval_size
        self.predictor = load_model(model_dir, use_gpu)

    def preprocess(self, image):
        """
        preprocess image: hwc_rgb to chw_bgr
        """
        img_mat = cv2.resize(
            image, self.eval_size, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        # HWC -> CHW
        img_mat = img_mat.swapaxes(1, 2)
        img_mat = img_mat.swapaxes(0, 1)
        # Convert to float
        img_mat = img_mat[:, :, :].astype('float32')
        # img_mat = (img_mat - mean) * scale
        img_mat = img_mat - self.mean
        img_mat = img_mat * self.scale
        img_mat = img_mat[np.newaxis, :, :, :]
        return img_mat

    def postprocess(self, image, output_data):
        """
        postprocess result: merge background with segmentation result
        """
        mask = output_data[0, 1, :, :]
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        scoremap = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        bg_im = np.ones_like(scoremap) * 255
        merge_im = (scoremap * image + (1 - scoremap) * bg_im).astype(np.uint8)
        return merge_im

    def run_predict(self, image):
        """
        run predict: return segmentation image mat
        """
        ori_im = image.copy()
        im_mat = self.preprocess(ori_im)
        im_tensor = fluid.core.PaddleTensor(im_mat.copy().astype('float32'))
        output_data = self.predictor.run([im_tensor])[0]
        output_data = output_data.as_ndarray()
        return self.postprocess(image, output_data)


def predict_image(seg, image_path):
    """
    Do Predicting on a image
    """
    img_mat = cv2.imread(image_path)
    img_mat = seg.run_predict(img_mat)
    cv2.imwrite('result.jpeg', img_mat)


def predict_video(seg, video_path):
    """
    Do Predicting on a video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Result Video Writer
    out = cv2.VideoWriter('result.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                          (width, height))
    # Start capturing from video
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img_mat = seg.run_predict(frame)
            out.write(img_mat)
        else:
            break
    cap.release()
    out.release()

def predict_camera(seg):
    """
    Do Predicting on a camera video stream
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    # Start capturing from video
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img_mat = seg.run_predict(frame)
            cv2.imshow('Frame', img_mat)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()

def main(argv):
    """
    Entrypoint of the script
    """
    if len(argv) < 3:
        print('Usage: python infer.py /path/to/model/ /path/to/video')
        return

    model_dir = sys.argv[1]
    input_path = sys.argv[2]
    use_gpu = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
    # Init model
    mean = [104.008, 116.669, 122.675]
    scale = [1.0, 1.0, 1.0]
    eval_size = (192, 192)
    seg = HumanSeg(model_dir, mean, scale, eval_size, use_gpu)
    # Run Predicting on a video and result will be saved as result.avi
    # predict_camera(seg)
    predict_video(seg, input_path)


if __name__ == "__main__":
    main(sys.argv)

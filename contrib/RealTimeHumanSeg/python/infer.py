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


def LoadModel(model_dir, use_gpu=False):
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
    def __init__(self, model_dir, mean, scale, eval_size, use_gpu=False):
        self.mean = np.array(mean).reshape((3, 1, 1))
        self.scale = np.array(scale).reshape((3, 1, 1))
        self.eval_size = eval_size
        self.predictor = LoadModel(model_dir, use_gpu)

    def Preprocess(self, image):
        im = cv2.resize(image,
                        self.eval_size,
                        fx=0,
                        fy=0,
                        interpolation=cv2.INTER_LINEAR)
        # HWC -> CHW
        im = im.swapaxes(1, 2)
        im = im.swapaxes(0, 1)
        # Convert to float
        im = im[:, :, :].astype('float32')
        # im  = (im - mean) * scale
        im = im - self.mean
        im = im * self.scale
        im = im[np.newaxis, :, :, :]
        return im

    def Postprocess(self, image, output_data):
        mask = output_data[0, 1, :, :]
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        scoremap = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        bg = np.ones_like(scoremap) * 255
        merge_im = (scoremap * image + (1 - scoremap) * bg).astype(np.uint8)
        return merge_im

    def Predict(self, image):
        ori_im = image.copy()
        im = self.Preprocess(image)
        im_tensor = fluid.core.PaddleTensor(im.copy().astype('float32'))
        output_data = self.predictor.run([im_tensor])[0]
        output_data = output_data.as_ndarray()
        return self.Postprocess(image, output_data)

# Do Predicting on a image
def PredictImage(seg, image_path):
    im = cv2.imread(input_path)
    im = seg.Predict(im)
    cv2.imwrite('result.jpeg', im)

# Do Predicting on a video
def PredictVideo(seg, video_path):
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened() == False:
        print("Error opening video stream or file")
        return
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Result Video Writer
    out = cv2.VideoWriter('result.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          fps,
                          (int(w), int(h)))
    # Start capturing from video
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            im = seg.Predict(frame)
            out.write(im)
        else:
            break
    cap.release()
    out.release()

# Do Predicting on a camera video stream
def PredictCamera(seg):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    # Start capturing from video
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            im = seg.Predict(frame)
            cv2.imshow('Frame', im)
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

    model_dir = argv[1]
    input_path = argv[2]
    use_gpu = int(argv[3]) if len(argv) >= 4 else 0
    # Init model
    mean = [104.008, 116.669, 122.675]
    scale = [1.0, 1.0, 1.0]
    eval_size = (192, 192)
    seg = HumanSeg(model_dir, mean, scale, eval_size, use_gpu)
    # Run Predicting on a video and result will be saved as result.avi
    PredictCamera(seg)


if __name__ == "__main__":
    main(sys.argv)

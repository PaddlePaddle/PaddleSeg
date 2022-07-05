import argparse
import os
import numpy as np
import cv2
from paddleseg.transforms import Normalize, Compose
from paddleseg.utils import get_image_list
from paddleseg.utils.visualize import get_pseudo_color_map
from onnxruntime import InferenceSession


def parse_args():
    parser = argparse.ArgumentParser(description='infer by onnxruntime')
    parser.add_argument(
        '--img_path',
        dest='img_path',
        help='The directory or path or file list of the images to be predicted.',
        type=str,
        default=None,
        required=True)
    parser.add_argument(
        '--onnx_file',
        dest='onnx_file',
        help='The onnx model path.',
        type=str,
        default=None,
        required=True)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the predict result.',
        type=str,
        default='./output')

    return parser.parse_args()


def _save_imgs(results, imgs_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, result in enumerate(results):
        if result.ndim == 3:
            result = np.squeeze(result)
        result = get_pseudo_color_map(result)
        basename = os.path.basename(imgs_path[i])
        basename, _ = os.path.splitext(basename)
        basename = f'{basename}.png'
        save_path = os.path.join(save_dir, basename)
        result.save(save_path)
        print('Predicted image is saved in {}'.format(save_path))


def main(args):
    img = cv2.imread(args.img_path)
    transform = Compose([Normalize()])
    input = transform(img)[0]
    input = input[np.newaxis, ...]

    # 加载ONNX模型
    sess = InferenceSession(args.onnx_file)

    # 模型预测
    ort_outs = sess.run(output_names=None,
                        input_feed={sess.get_inputs()[0].name: input})

    _save_imgs(ort_outs, [args.img_path], args.save_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)

import argparse
import os
import os.path as osp
import cv2
import numpy as np

from utils.humanseg_postprocess import postprocess
import models
import transforms


def parse_args():
    parser = argparse.ArgumentParser(description='HumanSeg inference for video')
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        help='Model path for inference',
        type=str)
    parser.add_argument(
        '--video_path',
        dest='video_path',
        help=
        'Video path for inference, camera will be used if the path not existing',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the inference results',
        type=str,
        default='./output')

    return parser.parse_args()


def video_infer(args):
    test_transforms = transforms.Compose(
        [transforms.Resize((192, 192)),
         transforms.Normalize()])
    model = models.load_model(args.model_dir)
    if not args.video_path:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise IOError("Error opening video stream or file, "
                      "--video_path whether existing: {}"
                      " or camera whether working".format(args.video_path))
        return
    if args.video_path:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 用于保存预测结果视频
        out = cv2.VideoWriter(
            osp.join(args.save_dir, 'result.avi'),
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))
        # 开始获取视频帧
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                results = model.predict(frame, test_transforms)
                img_mat = postprocess(frame, results['score_map'])
                out.write(img_mat)
            else:
                break
        cap.release()
        out.release()

    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                results = model.predict(frame, test_transforms)
                print(frame.shape, results['score_map'].shape)
                img_mat = postprocess(frame, results['score_map'])
                cv2.imshow('HumanSegmentation', img_mat)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()


if __name__ == "__main__":
    args = parse_args()
    video_infer(args)

import sys
import os
import numpy as np
import argparse

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))

from paddle_serving_client import Client
from paddle_serving_app.reader import Sequential, File2Image, Resize, CenterCrop
from paddle_serving_app.reader import RGB2BGR, Transpose, Div, Normalize

from paddleseg.utils.visualize import get_pseudo_color_map


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        "--serving_client_path",
        help="The path of serving_client file.",
        type=str,
        required=True)
    parser.add_argument(
        "--serving_ip_port",
        help="The serving ip.",
        type=str,
        default="127.0.0.1:9292",
        required=True)
    parser.add_argument(
        "--image_path", help="The image path.", type=str, required=True)
    return parser.parse_args()


def run(args):
    client = Client()
    client.load_client_config(
        os.path.join(args.serving_client_path, "serving_client_conf.prototxt"))
    client.connect([args.serving_ip_port])

    seq = Sequential([
        File2Image(), RGB2BGR(), Div(255),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], False), Transpose((2, 0, 1))
    ])

    img = seq(args.image_path)
    fetch_map = client.predict(
        feed={"x": img}, fetch=["save_infer_model/scale_0.tmp_1"])

    result = fetch_map["save_infer_model/scale_0.tmp_1"]
    color_img = get_pseudo_color_map(result[0])
    color_img.save("./result.png")
    print("The segmentation image is saved in ./result.png")


if __name__ == '__main__':
    args = parse_args()
    run(args)

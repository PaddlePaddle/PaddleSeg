import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Path of PaddleSeg model.")
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Type of inference device, support 'kunlunxin', 'cpu' or 'gpu'.")
    parser.add_argument(
        "--use_trt",
        type=ast.literal_eval,
        default=False,
        help="Wether to use tensorrt.")
    return parser.parse_args()


def build_option(args):
    option = fd.RuntimeOption()

    if args.device.lower() == "gpu":
        option.use_gpu()

    if args.use_trt:
        option.use_trt_backend()
        # If use original Tensorrt, not Paddle-TensorRT,
        # comment the following two lines
        option.enable_paddle_to_trt()
        option.enable_paddle_trt_collect_shape()
        option.set_trt_input_shape("x", [1, 3, 256, 256], [1, 3, 1024, 1024],
                                   [1, 3, 2048, 2048])
    return option


args = parse_arguments()

# settting for runtime
runtime_option = build_option(args)
model_file = os.path.join(args.model, "model.pdmodel")
params_file = os.path.join(args.model, "model.pdiparams")
config_file = os.path.join(args.model, "deploy.yaml")
model = fd.vision.segmentation.PaddleSegModel(
    model_file, params_file, config_file, runtime_option=runtime_option)

# predict
im = cv2.imread(args.image)
result = model.predict(im)
print(result)

# visualize
vis_im = fd.vision.vis_segmentation(im, result, weight=0.5)
cv2.imwrite("vis_img.png", vis_im)

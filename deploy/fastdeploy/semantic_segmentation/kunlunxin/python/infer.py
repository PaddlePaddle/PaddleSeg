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
    return parser.parse_args()


runtime_option = fd.RuntimeOption()
runtime_option.use_kunlunxin()

# setup runtime
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

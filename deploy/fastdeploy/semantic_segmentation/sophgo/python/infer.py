import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path of model.")
    parser.add_argument(
        "--config_file", required=True, help="Path of config file.")
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")

    return parser.parse_args()


args = parse_arguments()

# 配置runtime，加载模型
runtime_option = fd.RuntimeOption()
runtime_option.use_sophgo()

model_file = args.model
params_file = ""
config_file = args.config_file

model = fd.vision.segmentation.PaddleSegModel(
    model_file,
    params_file,
    config_file,
    runtime_option=runtime_option,
    model_format=fd.ModelFormat.SOPHGO)

# 预测图片分类结果
im_org = cv2.imread(args.image)
#bmodel 是静态模型，模型输入固定，这里设置为[512, 512]
im = cv2.resize(im_org, [512, 512], interpolation=cv2.INTER_LINEAR)
result = model.predict(im)
print(result)

# 预测结果可视化
vis_im = fd.vision.vis_segmentation(im, result, weight=0.5)
cv2.imwrite("sophgo_img.png", vis_im)

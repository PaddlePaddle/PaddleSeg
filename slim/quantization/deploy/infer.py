import ast
import os
import time

import argparse
import cv2
import yaml
import numpy as np
import paddle.fluid as fluid


class DeployConfig:
    def __init__(self, conf_file):
        if not os.path.exists(conf_file):
            raise Exception('Config file path [%s] invalid!' % conf_file)

        with open(conf_file) as fp:
            configs = yaml.load(fp, Loader=yaml.FullLoader)
            deploy_conf = configs["DEPLOY"]
            # 1. get eval_crop_size
            self.eval_crop_size = ast.literal_eval(
                deploy_conf["EVAL_CROP_SIZE"])
            # 2. get mean
            self.mean = deploy_conf["MEAN"]
            # 3. get std
            self.std = deploy_conf["STD"]
            # 4. get class_num
            self.class_num = deploy_conf["NUM_CLASSES"]
            # 5. get paddle model and params file path
            self.model_file = os.path.join(deploy_conf["MODEL_PATH"],
                                           deploy_conf["MODEL_FILENAME"])
            self.param_file = os.path.join(deploy_conf["MODEL_PATH"],
                                           deploy_conf["PARAMS_FILENAME"])
            # 6. use_gpu
            self.use_gpu = deploy_conf["USE_GPU"]
            # 7. predictor_mode
            self.predictor_mode = deploy_conf["PREDICTOR_MODE"]
            # 8. batch_size
            self.batch_size = deploy_conf["BATCH_SIZE"]
            # 9. channels
            self.channels = deploy_conf["CHANNELS"]


def create_predictor(args):
    predictor_config = fluid.core.AnalysisConfig(args.conf.model_file,
                                                 args.conf.param_file)
    predictor_config.enable_use_gpu(100, 0)
    predictor_config.switch_ir_optim(True)
    precision_type = fluid.core.AnalysisConfig.Precision.Float32 if not args.use_int8 else fluid.core.AnalysisConfig.Precision.Int8
    use_calib = False
    predictor_config.enable_tensorrt_engine(
        workspace_size=1 << 30,
        max_batch_size=1,
        min_subgraph_size=40,
        precision_mode=precision_type,
        use_static=False,
        use_calib_mode=use_calib)
    predictor_config.switch_specify_input_names(True)
    predictor_config.enable_memory_optim()
    predictor = fluid.core.create_paddle_predictor(predictor_config)

    return predictor


def preprocess(conf, image_path):
    flag = cv2.IMREAD_UNCHANGED if conf.channels == 4 else cv2.IMREAD_COLOR
    im = cv2.imread(image_path, flag)

    channels = im.shape[2]
    if channels != 3 and channels != 4:
        print('Only support rgb(gray) or rgba image.')
        return -1

    ori_h = im.shape[0]
    ori_w = im.shape[1]
    eval_w, eval_h = conf.eval_crop_size
    if ori_h != eval_h or ori_w != eval_w:
        im = cv2.resize(
            im, (eval_w, eval_h), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)

    im_mean = np.array(conf.mean).reshape((conf.channels, 1, 1))
    im_std = np.array(conf.std).reshape((conf.channels, 1, 1))

    im = im.swapaxes(1, 2)
    im = im.swapaxes(0, 1)
    im = im[:, :, :].astype('float32') / 255.0
    im -= im_mean
    im /= im_std

    im = im[np.newaxis, :, :, :]
    info = [image_path, im, (ori_w, ori_h)]
    return info


# Generate ColorMap for visualization
def generate_colormap(num_classes):
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map


def infer(args):
    predictor = create_predictor(args)
    colormap = generate_colormap(args.conf.class_num)

    images = get_images_from_dir(args.input_dir, args.ext)

    for image in images:
        im_info = preprocess(args.conf, image)

        input_tensor = fluid.core.PaddleTensor()
        input_tensor.name = 'image'
        input_tensor.shape = im_info[1].shape
        input_tensor.dtype = fluid.core.PaddleDType.FLOAT32
        input_tensor.data = fluid.core.PaddleBuf(
            im_info[1].ravel().astype("float32"))
        input_tensor = [input_tensor]

        output_tensor = predictor.run(input_tensor)[0]
        output_data = output_tensor.as_ndarray()

        img_name = im_info[0]
        ori_shape = im_info[2]

        logit = np.argmax(output_data, axis=1).squeeze()
        logit = logit.astype('uint8')[:, :, np.newaxis]
        logit = np.concatenate([logit] * 3, axis=2)

        for i in range(logit.shape[0]):
            for j in range(logit.shape[1]):
                logit[i, j] = colormap[logit[i, j, 0]]

        logit = cv2.resize(
            logit, ori_shape, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

        save_path = os.path.join(args.save_dir, img_name)
        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        cv2.imwrite(save_path, logit, [cv2.CV_8UC1])


def get_images_from_dir(img_dir, support_ext='.jpg|.jpeg'):
    if (not os.path.exists(img_dir) or not os.path.isdir(img_dir)):
        raise Exception('Image Directory [%s] invalid' % img_dir)
    imgs = []
    for item in os.listdir(img_dir):
        ext = os.path.splitext(item)[1][1:].strip().lower()
        if (len(ext) > 0 and ext in support_ext):
            item_path = os.path.join(img_dir, item)
            imgs.append(item_path)
    return imgs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--conf', type=str, default='', help='Configuration File Path.')
    parser.add_argument(
        '--ext',
        type=str,
        default='.jpeg|.jpg',
        help='Input Image File Extensions.')
    parser.add_argument(
        '--use_int8',
        dest='use_int8',
        action='store_true',
        help='Whether to use int8 for prediction.')
    parser.add_argument(
        '--input_dir',
        type=str,
        help='Directory that store images to be predicted.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='output',
        help='Directory for saving the predict results.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.conf = DeployConfig(args.conf)
    result = infer(args)

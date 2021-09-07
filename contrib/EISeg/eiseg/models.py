import os.path as osp

import paddle

from model.is_hrnet_model import HRNetModel
# from model.is_deeplab_model import DeeplabModel
# from util import model_path

here = osp.dirname(osp.abspath(__file__))


class HRNet18_OCR48:
    name = "HRNet18_OCR48"

    def load_params(self, params_path):
        model = HRNetModel(
            width=18, 
            ocr_width=48, 
            small=True, 
            with_aux_output=True, 
            use_rgb_conv=False,
            use_leaky_relu=True, 
            use_disks=True, 
            with_prev_mask=True, 
            norm_radius=5, 
            cpu_dist_maps=False
        )
        para_state_dict = paddle.load(params_path)
        model.set_dict(para_state_dict)
        model.eval()
        return model


class HRNet18_OCR64:
    name = "HRNet18_OCR64"

    def load_params(self, params_path):
        model = HRNetModel(
            width=18, 
            ocr_width=64, 
            small=False, 
            with_aux_output=True, 
            use_leaky_relu=True,
            use_rgb_conv=False,
            use_disks=True, 
            norm_radius=5, 
            with_prev_mask=True,
            cpu_dist_maps=False  # 目前打包cython有些问题，先默认用False
        )
        para_state_dict = paddle.load(params_path)
        model.set_dict(para_state_dict)
        model.eval()
        return model


models = [HRNet18_OCR48(), HRNet18_OCR64()]


def findModelbyName(model_name):
    for idx, mt in enumerate(models):
        if model_name == mt.name:
            return models[idx], idx
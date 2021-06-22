import os.path as osp

import paddle

from model.is_hrnet_model import HRNetModel
from model.is_deeplab_model import DeeplabModel
# from util import model_path

here = osp.dirname(osp.abspath(__file__))


class HRNet:
    name = "HRNetModel"

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
            cpu_dist_maps=False,
        )
        para_state_dict = paddle.load(params_path)
        model.set_dict(para_state_dict)
        return model

class Deeplab:
    name = "DeeplabModel"
    # 下面的参数没有进行调整
    def load_params(self, params_path):
        model = DeeplabModel(
            width=18,
            ocr_width=48,
            small=True,
            with_aux_output=True,
            use_rgb_conv=False,
            use_leaky_relu=True,
            use_disks=True,
            with_prev_mask=True,
            norm_radius=5,
            cpu_dist_maps=False,
        )
        para_state_dict = paddle.load(params_path)
        model.set_dict(para_state_dict)
        return model


models = [HRNet(), Deeplab()]

import os.path as osp
from abc import ABC, abstractmethod

import paddle

from model.is_hrnet_model import HRNetModel
from util import MODELS

here = osp.dirname(osp.abspath(__file__))


class EISegModel:
    @abstractmethod
    def __init__(self):
        self.paramSet = False
        try:
            self.create_model()
        except AssertionError:
            ver = paddle.__version__
            if ver < "2.1.0":
                raise Exception("模型创建失败。Paddle版本低于2.1.0，请升级paddlepaddle")
            else:
                raise Exception("模型创建失败。请参考官网教程检查Paddle安装是否正确，GPU版本请注意是否正确安装显卡驱动。")

    def load_param(self, param_path):
        params = self.get_param(param_path)
        if params:
            try:
                self.model.set_dict(params)
                self.model.eval()
            except:
                raise Exception("权重设置失败。请参考官网教程检查Paddle安装是否正确，GPU版本请注意是否正确安装显卡驱动。")
            self.paramSet = True
            return True
        else:
            return None

    def get_param(self, param_path):
        print("param_path", self.__name__, param_path)
        if param_path is None or not osp.exists(param_path):
            raise Exception(f"权重路径{param_path}不存在。请指定正确的模型路径")
        params = paddle.load(param_path)
        pkeys = params.keys()
        mkeys = self.model.named_parameters()
        if len(pkeys) != len(list(mkeys)):
            raise Exception("权重和模型不匹配。请确保指定的权重和模型对应")
        for p, m in zip(pkeys, mkeys):
            if p != m[0]:
                raise Exception("权重和模型不匹配。请确保指定的权重和模型对应")
        return params


ModelsNick = {"HRNet18s_OCR48": ["轻量级模型", 0],
              "HRNet18_OCR64": ["高精度模型", 1]}

@MODELS.add_component
class HRNet18s_OCR48(EISegModel):
    __name__ = "HRNet18s_OCR48"

    def create_model(self):
        self.model = HRNetModel(
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


@MODELS.add_component
class HRNet18_OCR64(EISegModel):
    __name__ = "HRNet18_OCR64"

    def create_model(self):
        self.model = HRNetModel(
            width=18,
            ocr_width=64,
            small=False,
            with_aux_output=True,
            use_leaky_relu=True,
            use_rgb_conv=False,
            use_disks=True,
            norm_radius=5,
            with_prev_mask=True,
            cpu_dist_maps=False,  # 目前打包cython有些问题，先默认用False
        )

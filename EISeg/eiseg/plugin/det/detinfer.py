from .detection_model import Detector
from util.coco.detlabel import COCO_CLASS_DICT


class DetInfer:
    def __init__(self, model_path=None):
        self.model = None if model_path is None else Detector(model_path)

    def load_model(self, model_path):
        self.model = Detector(model_path)

    def infer(self, img):
        if self.model is None:
            # 默认在没有模型得情况下使用推理时直接返回空列表
            return []
        output = self.model.run(img, visual=False)
        res_list = []
        for anchor in output["boxes"]:
            clas_id, bbox, score = int(anchor[0]), anchor[2:], anchor[1]
            if score > self.model.threshold:
                x_min, y_min, x_max, y_max = bbox
                p = (int(x_min), int(y_min))
                size = (int(x_max - x_min), int(y_max - y_min))
                res_list.append((COCO_CLASS_DICT[clas_id], p, size))
        return res_list

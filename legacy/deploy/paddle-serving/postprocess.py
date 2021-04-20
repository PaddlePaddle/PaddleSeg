import cv2
import numpy as np


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


class SegPostprocess(object):
    def __init__(self, class_num):
        self.class_num = class_num

    def __call__(self, image_with_result):
        if "filename" not in image_with_result:
            raise ("filename should be specified in postprocess")
        img_name = image_with_result["filename"]
        ori_img = cv2.imread(img_name, -1)
        ori_shape = ori_img.shape
        mask = None
        for key in image_with_result:
            if ".lod" in key or "filename" in key:
                continue
            mask = image_with_result[key][0]
        if mask is None:
            raise ("segment mask should be specified in postprocess")
        mask = np.argmax(mask, axis=0)
        mask = mask.astype("uint8")
        mask_png = mask
        score_png = mask_png[:, :, np.newaxis]
        score_png = np.concatenate([score_png] * 3, axis=2)

        color_map = generate_colormap(self.class_num)
        for i in range(score_png.shape[0]):
            for j in range(score_png.shape[1]):
                score_png[i, j] = color_map[score_png[i, j, 0]]
        ext_pos = img_name.rfind(".")
        img_name_fix = img_name[:ext_pos] + "_" + img_name[ext_pos + 1:]
        mask_save_name = img_name_fix + "_mask.png"
        cv2.imwrite(mask_save_name, mask_png, [cv2.CV_8UC1])
        vis_result_name = img_name_fix + "_result.png"
        result_png = score_png

        result_png = cv2.resize(
            result_png, (ori_shape[1], ori_shape[0]),
            fx=0,
            fy=0,
            interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(vis_result_name, result_png, [cv2.CV_8UC1])
        return result_png

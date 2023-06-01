import os
import os.path as osp
from PIL import Image
import json
import numpy as np
from eiseg.util import get_polygon, COCO, colorMap
from eiseg.util.jsencoder import JSEncoder


def getPolygonFromMask(mask):
    m_shape = mask.shape
    polygon = []
    umm = np.unique(mask)
    if len(umm) == 2 and max(umm) == 255:
        mask = np.clip(mask, 0, 1)
    for i_clas in umm:
        if i_clas == 0:
            continue
        tmp_mask = (mask == i_clas).astype("uint8") * 255
        polygon.append({
            "label": i_clas,
            "points": get_polygon(tmp_mask, img_size=m_shape)
        })
    return polygon


def saveMaskToCOCO(image_dir, mask_dir):
    img_names = sorted(os.listdir(image_dir))
    mask_names = sorted(os.listdir(mask_dir))
    if (len(img_names) != len(mask_names)):
        return
    label_dir = osp.join(image_dir, "label")
    if not osp.exists(label_dir):
        os.makedirs(label_dir)
    coco_path = osp.join(label_dir, "annotations.json")
    if osp.exists(coco_path):
        os.remove(coco_path)
    coco = COCO(coco_path)
    labelList = []
    for img_name, mask_name in zip(img_names, mask_names):
        img_path = osp.join(image_dir, img_name)
        mask_path = osp.join(mask_dir, mask_name)
        mask = np.asarray(Image.open(mask_path).convert('L')).astype("uint8")
        imgId = coco.addImage(osp.basename(img_path), mask.shape[1], mask.shape[0])
        polygons = getPolygonFromMask(mask)
        for polygon in polygons:
            coco_id = None
            labelIndex = polygon["label"]
            for points in polygon["points"]:
                new_points = []
                for p in points:
                    new_points.extend([p[0], p[1]])
                if not coco_id:
                    annId = coco.addAnnotation(imgId, labelIndex, new_points)
                    coco_id = annId
                else:
                    coco.updateAnnotation(coco_id, imgId, new_points)
            labelList.append(labelIndex)
    labelList = list(set(labelList))
    labelList.sort()
    for lab in labelList:
        color = colorMap.colors[lab]
        if coco.hasCat(lab):
            coco.updateCategory(lab, str(lab), color)
        else:
            coco.addCategory(lab, str(lab), color)
    open(coco_path, "w", encoding="utf-8").write(json.dumps(coco.dataset, cls=JSEncoder))

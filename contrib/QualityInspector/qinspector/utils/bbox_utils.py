import numpy as np
from PIL import Image, ImageDraw


def square(bbox, img_shape):
    """Calculate square bounding box 

    Args:
        bbox (list):[x1, y1, x2, y2]
        img_shape (tuple): (height, width)

    Returns:
        bbox (list)
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1 + 1, y2 - y1 + 1
    if w < h:
        pad = (h - w) // 2
        x1 = max(0, x1 - pad)
        x2 = min(img_shape[1], x2 + pad)
    else:
        pad = (w - h) // 2
        y1 = max(0, y1 - pad)
        y2 = min(img_shape[0], y2 + pad)
    return [x1, y1, x2, y2]


def padding(bbox, img_shape, pad_scale=0.0):
    """pad bbox with scale

    Args:
        bbox (list):[x1, y1, x2, y2]
        img_shape (tuple): (height, width)
        pad_scale (float): scale for padding

    Returns:
        bbox (list)
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1 + 1, y2 - y1 + 1
    dw = int(w * pad_scale)
    dh = int(h * pad_scale)
    x1 = max(0, x1 - dw)
    x2 = min(img_shape[1], x2 + dw)
    y1 = max(0, y1 - dh)
    y2 = min(img_shape[0], y2 + dh)
    return [int(x1), int(y1), int(x2), int(y2)]


def adjust_bbox(bbox, img_shape, pad_scale=0.0):
    """adjust box according to img_shape and pad_scale 

    Args:
        bbox (list):[x1, y1, x2, y2]
        img_shape (tuple): (height, width)
        pad_scale (float): scale for padding

    Returns:
        bbox (list)
    """
    bbox = square(bbox, img_shape)
    bbox = padding(bbox, img_shape, pad_scale)
    return bbox


def iou_one_to_multiple(box, boxes):
    """
    Calculate the Intersection over Union (IoU) of a bounding box with a batch of bounding boxes.

    Args:
        box (list of 4 floats): [xmin, ymin, w, h]
        boxes (list of N lists of 4 floats): [[xmin, ymin, w, h], [xmin, ymin, w, h], ...]

    Returns:
        list of N floats: the IoU of the box with each of the boxes in the batch
    """
    boxes = np.array(boxes, dtype=np.float32).reshape([-1, 4])
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2])
    y2 = np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3])
    intersection_area = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
    box_area = box[2] * box[3]
    boxes_area = boxes[:, 2] * boxes[:, 3]
    union_area = box_area + boxes_area - intersection_area
    iou = intersection_area / union_area
    return iou


def get_bbox(height, width, points):
    """polygon to box

    Args:
        height (int): height of image
        width (int): width of image
        points (list): poly point [x, y]

    Returns:
        bbox (list): [x1, y1, w, h]
    """
    polygons = points
    mask = np.zeros([height, width], dtype=np.uint8)
    mask = Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    index = np.argwhere(mask == 1)
    rows = index[:, 0]
    clos = index[:, 1]
    left_top_r = np.min(rows)
    left_top_c = np.min(clos)
    right_bottom_r = np.max(rows)
    right_bottom_c = np.max(clos)
    return [
        left_top_c, left_top_r, right_bottom_c - left_top_c,
        right_bottom_r - left_top_r
    ]
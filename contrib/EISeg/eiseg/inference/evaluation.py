from time import time
import numpy as np
import paddle
from .utils import *
from inference.clicker import Clicker
from tqdm import tqdm
import scipy.misc as sm
import os


def evaluate_dataset(dataset, predictor, oracle_eval=False, **kwargs):
    all_ious = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)
        _, sample_ious, _ = evaluate_sample(sample.image, sample.gt_mask, predictor,
                                            sample_id=index, **kwargs)
        all_ious.append(sample_ious)

    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time


def evaluate_sample(image, gt_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, callback=None):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []

    predictor.set_input_image(image)
    pred_probs = None

    for click_indx in range(max_clicks):
        clicker.make_next_click(pred_mask)
        pred_probs = predictor.get_prediction(clicker)
        pred_mask = pred_probs > pred_thr

        if callback is not None:
            callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

        iou = get_iou(gt_mask, pred_mask)
        ious_list.append(iou)

        if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
            # image_name = str(time()) + '.png'
            # sm.imsave(os.path.join('result', image_name), pred_probs)
            break

    return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs

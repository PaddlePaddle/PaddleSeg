from datetime import timedelta
from pathlib import Path
import paddle

import numpy as np

from data.datasets import GrabCutDataset, BerkeleyDataset, DavisDataset, SBDEvaluationDataset, PascalVocDataset, HumanDataset
from util.serialization import load_model


def get_time_metrics(all_ious, elapsed_time):
    n_images = len(all_ious)
    n_clicks = sum(map(len, all_ious))

    mean_spc = elapsed_time / n_clicks
    mean_spi = elapsed_time / n_images

    return mean_spc, mean_spi


def load_is_model(checkpoint, **kwargs):
    if isinstance(checkpoint, (str, Path)):
        state_dict = paddle.load(checkpoint)
    else:
        state_dict = checkpoint

    if isinstance(state_dict, list):
        model = load_single_is_model(state_dict[0],  **kwargs)
        models = [load_single_is_model(x, **kwargs) for x in state_dict]

        return model, models
    else:
        return load_single_is_model(state_dict, **kwargs)


def load_single_is_model(state_dict, **kwargs):
    model = load_model(state_dict['config'], **kwargs)
    model.load_state_dict(state_dict['state_dict'], strict=False)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model


def get_dataset(dataset_name, cfg):
    if dataset_name == 'GrabCut':
        dataset = GrabCutDataset(cfg.GRABCUT_PATH)
    elif dataset_name == 'Berkeley':
        dataset = BerkeleyDataset(cfg.BERKELEY_PATH)
    elif dataset_name == 'DAVIS':
        dataset = DavisDataset(cfg.DAVIS_PATH)
    elif dataset_name == 'SBD':
        dataset = SBDEvaluationDataset(cfg.SBD_PATH)
    elif dataset_name == 'SBD_Train':
        dataset = SBDEvaluationDataset(cfg.SBD_PATH, split='train')
    elif dataset_name == 'PascalVOC':
        dataset = PascalVocDataset(cfg.PASCALVOC_PATH, split='val')
    elif dataset_name == 'COCO_MVal':
        dataset = DavisDataset(cfg.COCO_MVAL_PATH)
    elif dataset_name == 'Human':
        dataset = HumanDataset(cfg.HUMAN_PATH, split='test')
    else:
        dataset = None

    return dataset


def get_iou(gt_mask, pred_mask, ignore_label=-1):
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = np.logical_and(np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = np.logical_and(np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

    return intersection / union


def compute_noc_metric(all_ious, iou_thrs, max_clicks=20):
    def _get_noc(iou_arr, iou_thr):
        vals = iou_arr >= iou_thr
        return np.argmax(vals) + 1 if np.any(vals) else max_clicks

    noc_list = []
    over_max_list = []
    for iou_thr in iou_thrs:
        scores_arr = np.array([_get_noc(iou_arr, iou_thr)
                               for iou_arr in all_ious], dtype=np.int)

        score = scores_arr.mean()
        over_max = (scores_arr == max_clicks).sum()

        noc_list.append(score)
        over_max_list.append(over_max)

    return noc_list, over_max_list


def find_checkpoint(weights_folder, checkpoint_name):
    weights_folder = Path(weights_folder)
    if ':' in checkpoint_name:
        model_name, checkpoint_name = checkpoint_name.split(':')
        models_candidates = [x for x in weights_folder.glob(f'{model_name}*') if x.is_dir()]
        assert len(models_candidates) == 1
        model_folder = models_candidates[0]
    else:
        model_folder = weights_folder

    if checkpoint_name.endswith('.pth'):
        if Path(checkpoint_name).exists():
            checkpoint_path = checkpoint_name
        else:
            checkpoint_path = weights_folder / checkpoint_name
    else:
        model_checkpoints = list(model_folder.rglob(f'{checkpoint_name}*.pdparams'))
        assert len(model_checkpoints) == 1
        checkpoint_path = model_checkpoints[0]

    return str(checkpoint_path)


def get_results_table(noc_list, over_max_list, brs_type, dataset_name, mean_spc, elapsed_time,
                      n_clicks=20, model_name=None):
    table_header = (f'|{"BRS Type":^13}|{"Dataset":^11}|'
                    f'{"NoC@80%":^9}|{"NoC@85%":^9}|{"NoC@90%":^9}|'
                    f'{">="+str(n_clicks)+"@85%":^9}|{">="+str(n_clicks)+"@90%":^9}|'
                    f'{"SPC,s":^7}|{"Time":^9}|')
    row_width = len(table_header)

    header = f'Eval results for model: {model_name}\n' if model_name is not None else ''
    header += '-' * row_width + '\n'
    header += table_header + '\n' + '-' * row_width

    eval_time = str(timedelta(seconds=int(elapsed_time)))
    table_row = f'|{brs_type:^13}|{dataset_name:^11}|'
    table_row += f'{noc_list[0]:^9.2f}|'
    table_row += f'{noc_list[1]:^9.2f}|' if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f'{noc_list[2]:^9.2f}|' if len(noc_list) > 2 else f'{"?":^9}|'
    table_row += f'{over_max_list[1]:^9}|' if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f'{over_max_list[2]:^9}|' if len(noc_list) > 2 else f'{"?":^9}|'
    table_row += f'{mean_spc:^7.3f}|{eval_time:^9}|'

    return header, table_row
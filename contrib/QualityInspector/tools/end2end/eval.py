# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

import argparse
import json
import os
import os.path as osp
import sys
from collections import defaultdict

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)

import prettytable as pt
from pycocotools.coco import COCO

from qinspector.cvlib.configs import ConfigParser
from qinspector.ops.postprocess import PostProcess
from qinspector.utils.logger import setup_logger
from qinspector.utils.bbox_utils import iou_one_to_multiple
from qinspector.utils.visualizer import show_result, draw_bboxes

logger = setup_logger('Eval')


def get_args():
    parser = argparse.ArgumentParser(description='Eval')
    # Parameters
    parser.add_argument(
        '--input_path',
        type=str,
        help='The path of coco format json file for evaluation.',
        required=True)
    parser.add_argument(
        '--pred_path',
        type=str,
        help='The path of json file saving prediction results obtained by `tools/end2end/predict.py`.',
        required=True)
    parser.add_argument(
        '--image_root', type=str, default='', help='The directory of images.')
    parser.add_argument(
        '--config', type=str, default=None, help='The path of config file.')
    parser.add_argument(
        '--rules_eval',
        action='store_true',
        help='Whether or not to update rules for postprocess, defalut: False.')
    parser.add_argument(
        '--instance_level',
        action='store_false',
        help='Whether or not to eval by instance-wise, default: True.')
    parser.add_argument(
        '--iou_theshold',
        type=float,
        default=0.1,
        help='IoU threshold for instance-wise evalution to judge recall.')
    parser.add_argument(
        '--save_badcase',
        action='store_false',
        help='Whether or not to save badcase, defalt: True.')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output/badcase/',
        help='The directory for saving bascases.')
    return parser.parse_args()


def read_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def set_class_to_set(data_dict):
    for cls in data_dict.keys():
        data_dict[cls] = set(data_dict[cls])
    return data_dict


def eval_ng(gt_data,
            gt_ng_ids,
            preds_data,
            image_root='',
            instance_level=True,
            iou_theshold=0.1):
    ok_image_in_ng_num, ok_instance_in_ng_num, ng_instance_num = 0, 0, 0
    class_list = gt_data.getCatIds()
    clsid_to_name = gt_data.cats
    class_name = [clsid_to_name[i]['name'] for i in class_list]
    ng_class_num = defaultdict(int)
    ok_in_ng_class_num = defaultdict(int)
    escape_info_image = []
    escape_info_instance = defaultdict(list)
    for ng_id in gt_ng_ids:
        im_name = gt_data.loadImgs(ng_id)[0]['file_name']
        preds_info = preds_data[osp.join(image_root, im_name)]
        if not preds_info['isNG']:
            ok_image_in_ng_num += 1
            escape_info_image.append(ng_id)
        if instance_level:
            gt_anno_id = gt_data.getAnnIds(imgIds=ng_id, iscrowd=False)
            gt_annos = gt_data.loadAnns(gt_anno_id)
            if len(gt_annos) == 0:
                continue
            ng_instance_num += len(gt_annos)
            if not preds_info['isNG']:
                ok_instance_in_ng_num += len(gt_annos)
                for gt in gt_annos:
                    ok_in_ng_class_num[gt['category_id']] += 1
                    ng_class_num[gt['category_id']] += 1
                    escape_info_instance[gt['category_id']].append(
                        (ng_id, gt['id']))
            else:
                pred_bbox = []
                for pred in preds_info['pred']:
                    if pred['isNG']:
                        pred_bbox.append(pred['bbox'])

                for gt in gt_annos:
                    ng_class_num[gt['category_id']] += 1
                    max_iou = max(iou_one_to_multiple(gt['bbox'], pred_bbox))
                    if max_iou < iou_theshold:
                        ok_instance_in_ng_num += 1
                        ok_in_ng_class_num[gt['category_id']] += 1
                        escape_info_instance[gt['category_id']].append(
                            (ng_id, gt['id']))

    ng_num = len(gt_ng_ids)
    column_names = ["Image Level", "Total", "NG", "OK", "Escape"]
    table = pt.PrettyTable(column_names)
    table.add_row([
        "Lucky Result",
        ng_num,
        ng_num - ok_image_in_ng_num,
        ok_image_in_ng_num,
        "{:.2f}%".format(ok_image_in_ng_num / ng_num * 100)
        if ng_num > 0 else 0,
    ])
    logger.info("Result of Image-Level NG NG Evaluation:\n" + str(table))

    if instance_level:
        ng_class_num = [ng_class_num[cat] for cat in class_list]
        ok_in_ng_class_num = [ok_in_ng_class_num[cat] for cat in class_list]
        column_names = ["NG", "ALL", *class_name]
        table = pt.PrettyTable(column_names)
        table.add_row(["Total", ng_instance_num, * [x for x in ng_class_num]])
        table.add_row([
            "NG", ng_instance_num - ok_instance_in_ng_num, * [
                ng_class_num[i] - ok_in_ng_class_num[i]
                for i in range(len(ng_class_num))
            ]
        ])
        table.add_row(
            ["OK", ok_instance_in_ng_num, * [x for x in ok_in_ng_class_num]])
        table.add_row([
            " Escape ",
            "{:.2f}%".format(ok_instance_in_ng_num / ng_instance_num * 100)
            if ng_instance_num > 0 else 0, * [
                "{:.2f}%".format(ok_in_ng_class_num[i] / ng_class_num[i] * 100)
                for i in range(len(ng_class_num))
            ]
        ])
        logger.info("Result of Instance-Level NG Evaluation:\n" + str(table))

    return escape_info_image, escape_info_instance


def eval_ok(gt_data, gt_ok_ids, preds_data, image_root=''):
    ng_in_ok_num = 0
    ng_info = defaultdict(list)
    class_list = gt_data.getCatIds()
    clsid_to_name = gt_data.cats
    class_name = [clsid_to_name[i]['name'] for i in class_list]

    for ok_id in gt_ok_ids:
        im_name = gt_data.loadImgs(ok_id)[0]['file_name']
        preds_info = preds_data[osp.join(image_root, im_name)]
        if preds_info['isNG']:
            ng_in_ok_num += 1
            for pred in preds_info['pred']:
                if pred['isNG']:
                    ng_info[pred['category_id']].append(ok_id)

    ng_info = set_class_to_set(ng_info)
    ng_class_nums = [len(ng_info[cat]) for cat in class_list]
    ok_num = max(len(gt_ok_ids), 1)

    column_names = ["OK", "ALL", *class_name]
    table = pt.PrettyTable(column_names)
    table.add_row(
        ["Total", len(gt_ok_ids), *([len(gt_ok_ids)] * len(class_name))])
    table.add_row([
        "OK", len(gt_ok_ids) - ng_in_ok_num,
        * [len(gt_ok_ids) - x for x in ng_class_nums]
    ])
    table.add_row(["NG", ng_in_ok_num, *ng_class_nums])
    table.add_row([
        "Overkill", "{:.2f}%".format((ng_in_ok_num / ok_num) * 100),
        * ["{:.2f}%".format(x / ok_num * 100) for x in ng_class_nums]
    ])
    logger.info("OK Evaluation Result:\n" + str(table))

    return ng_info


def post_process_image_info(input):
    for _, img_info in input.items():
        preds = img_info['pred']
        img_info['isNG'] = 0
        if any((pred['isNG'] == 1) for pred in preds):
            img_info['isNG'] = 1
    return input


def evaluation(gt_data,
               preds_data,
               post_modules=None,
               image_root='',
               instance_level=True,
               iou_theshold=0.1):
    if post_modules:
        for _, img_info in preds_data.items():
            preds = img_info['pred']
            img_info.pop('isNG')
            for pred in preds:
                pred.pop('isNG')
        preds_data = post_modules(preds_data)
        post_process_image_info(preds_data)

    img_ids = list(sorted(gt_data.imgs.keys()))
    gt_ok_ids = []
    gt_ng_ids = []
    for img_id in img_ids:
        ann_ids = gt_data.getAnnIds(imgIds=img_id, iscrowd=False)
        annos = gt_data.loadAnns(ann_ids)
        if len(annos) == 0:
            gt_ok_ids.append(img_id)
        else:
            gt_ng_ids.append(img_id)

    overkill_info = eval_ok(
        gt_data, gt_ok_ids, preds_data,
        image_root)  # overkill_info = {class_id: set(img_id)}
    escape_info_image, escape_info_instance = eval_ng(
        gt_data,
        gt_ng_ids,
        preds_data,
        image_root,
        instance_level=instance_level,
        iou_theshold=iou_theshold)

    return overkill_info, escape_info_image, escape_info_instance


def show_badcase(gt_data,
                 preds_data,
                 overkill_info,
                 escape_info_image,
                 escape_info_instance,
                 image_root='',
                 output_dir=''):
    overkill_path = osp.join(output_dir, 'overkill')
    escape_image_path = osp.join(output_dir, 'escape', 'image_level')
    escape_instance_path = osp.join(output_dir, 'escape', 'instance_level')

    logger.info("Save overkill images in: " + overkill_path)
    clsid_to_name = gt_data.cats
    for class_id, img_ids in overkill_info.items():
        class_name = clsid_to_name[class_id]['name']
        for img_id in img_ids:
            im_name = gt_data.loadImgs(img_id)[0]['file_name']
            im_path = osp.join(image_root, im_name)
            preds_info = preds_data[im_path]
            show_result({
                im_path: preds_info
            }, osp.join(overkill_path, class_name), class_name)

    logger.info("Save escape images in: " + escape_image_path)
    for img_id in escape_info_image:
        im_name = gt_data.loadImgs(img_id)[0]['file_name']
        ann_ids = gt_data.getAnnIds(imgIds=img_id, iscrowd=None)
        annos = gt_data.loadAnns(ann_ids)
        im_path = osp.join(image_root, im_name)
        draw_bboxes(im_path, annos, clsid_to_name, escape_image_path)

    logger.info("Save instance-level escape images in: " + escape_instance_path)
    for class_id, img_anno_ids in escape_info_instance.items():
        class_name = clsid_to_name[class_id]['name']
        img_to_anno_id = defaultdict(list)
        for img_anno_id in img_anno_ids:
            img_id = img_anno_id[0]
            anno_id = img_anno_id[1]
            img_to_anno_id[img_id].append(anno_id)

        for img_id, anno_list in img_to_anno_id.items():
            im_name = gt_data.loadImgs(img_id)[0]['file_name']
            annos = gt_data.loadAnns(anno_list)
            im_path = osp.join(image_root, im_name)
            draw_bboxes(im_path, annos, clsid_to_name,
                        osp.join(escape_instance_path, class_name))
    logger.info("Done!")


def main():
    args = get_args()
    gt_data = COCO(args.input_path)
    img_to_pred_annos = read_json(args.pred_path)
    post_modules = None
    if args.rules_eval:
        config = ConfigParser(args)
        postprocess = config.parse()[0][-1]
        post_modules = PostProcess(postprocess['PostProcess'])

    overkill_info, escape_info_image, escape_info_instance = evaluation(
        gt_data, img_to_pred_annos, post_modules, args.image_root,
        args.instance_level, args.iou_theshold)

    if args.save_badcase:
        show_badcase(gt_data, img_to_pred_annos, overkill_info,
                     escape_info_image, escape_info_instance, args.image_root,
                     args.output_dir)


if __name__ == '__main__':
    main()

import numpy as np
import os
import subprocess
import json
import argparse
import glob

from val import evaluate


def init_args():
    parser = argparse.ArgumentParser()
    # params for testing assert allclose
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--metric_file", type=str, default="")
    parser.add_argument("--predict_dir", type=str, default="")
    parser.add_argument("--gt_dir", type=str, default="")
    parser.add_argument("--num_classes", type=int)
    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()


def run_shell_command(cmd):
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = p.communicate()

    if p.returncode == 0:
        return out.decode('utf-8')
    else:
        return None


def load_gt_from_file(metric_file):
    if not os.path.exists(metric_file):
        raise ValueError("The log file {} does not exists!".format(metric_file))
    with open(metric_file, 'r') as f:
        data = f.readlines()
        f.close()
    parser_gt = {}
    for line in data:
        metric, result = line.strip("\n").split(":")
        if 'Class' in metric:
            parser_gt[metric] = result
        else:
            parser_gt[metric] = float(result)
    return parser_gt


def load_metric_from_txts(metric_file):
    gt_list = glob.glob(metric_file)
    true_metrics = {}
    for gt_f in gt_list:
        gt_dict = load_gt_from_file(gt_f)
        basename = os.path.basename(gt_f)
        if "fp32" in basename:
            true_metrics["fp32"] = [gt_dict, gt_f]
        elif "fp16" in basename:
            true_metrics["fp16"] = [gt_dict, gt_f]
        elif "int8" in basename:
            true_metrics["int8"] = [gt_dict, gt_f]
        else:
            continue
    return true_metrics


def cal_metric(predict_dir, gt_dir, num_classes, key_list):
    predict_list = glob.glob(predict_dir)
    pred_metics = {}
    for predict_dir_ in predict_list:
        key = os.path.basename(predict_dir_)
        print(key)
        pred_dict = evaluate(predict_dir_, gt_dir, num_classes)
        pred_metics[key] = pred_dict
    return pred_metics


def testing_assert_allclose(dict_x, dict_y, atol=1e-7, rtol=1e-7):
    for k in dict_x:
        if 'Class' in k:
            continue
        np.testing.assert_allclose(
            np.array(dict_x[k]), np.array(dict_y[k]), atol=atol, rtol=rtol)


if __name__ == "__main__":
    # Usage:
    # python3.7 test_tipc/compare_results.py --metric_file=./test_tipc/results/*.txt  --predict_dir=./test_tipc/output/fcn_hrnetw18_small/python_infer_*_results --gt_dir=./test_tipc/data/mini_supervisely/Annotations --num_classes 2

    args = parse_args()

    true_metrics = load_metric_from_txts(args.metric_file)
    key_list = true_metrics["fp32"][0].keys()

    pred_metics = cal_metric(args.predict_dir, args.gt_dir, args.num_classes,
                             key_list)
    for filename in pred_metics.keys():
        if "fp32" in filename:
            gt_dict, gt_filename = true_metrics["fp32"]
        elif "fp16" in filename:
            gt_dict, gt_filename = true_metrics["fp16"]
        elif "int8" in filename:
            gt_dict, gt_filename = true_metrics["int8"]
        else:
            continue
        pred_dict = pred_metics[filename]

        try:
            testing_assert_allclose(
                gt_dict, pred_dict, atol=args.atol, rtol=args.rtol)
            print(
                "Assert allclose passed! The results of {} and {} are consistent!"
                .format(filename, gt_filename))
        except Exception as E:
            print(E)
            print(
                "Assert allclose failed! The results of {} and the results of {} are inconsistent!"
                .format(filename, gt_filename))

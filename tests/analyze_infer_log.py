import argparse
import re
import sys
"""
Read the log of `test_infer_benchmark.sh` and `test_infer_dataset.sh`, collect the accuracy and speed for inference models.

Usage: python analyze_infer_log.py --log_path /path/to/load/log --save_path /path/to/save/info
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze logs')
    parser.add_argument(
        "--log_path", help="The path of log file.", type=str, required=True)
    parser.add_argument(
        "--save_path",
        help="The path of log file.",
        default="info.txt",
        type=str)
    return parser.parse_args()


def analyze(log_path):
    results = {}
    logs = open(log_path).readlines()

    model_pattern = re.compile(r"Test (.*) (.*) (.*) (fp\d+)")
    miou_pattern = re.compile(r"mIoU: (0.\d+)")
    time_pattern = re.compile(r"Average time: (.*) ms/img")

    # collect the start num for each test
    model_line_num = []
    for i in range(len(logs)):
        match_obj = re.search(model_pattern, logs[i])
        if match_obj is not None:
            model_line_num.append(i)

    # collect the useful information for each test
    for i in range(len(model_line_num)):
        start_num = model_line_num[i]
        end_num = len(logs) - 1 if i == len(
            model_line_num) - 1 else model_line_num[i + 1] - 1

        match_obj = re.search(model_pattern, logs[start_num])
        assert match_obj is not None
        model_name = match_obj.group(1)
        info_name = match_obj.group(3) + "_" + match_obj.group(4)
        miou_value = None
        time_value = None

        cur_num = end_num
        while cur_num >= start_num and (miou_value is None or
                                        time_value is None):
            line = logs[cur_num]
            cur_num -= 1
            if miou_value is None:
                match_obj = re.search(miou_pattern, line)
                if match_obj is not None:
                    miou_value = float(match_obj.group(1))
            if time_value is None:
                match_obj = re.search(time_pattern, line)
                if match_obj is not None:
                    time_value = float(match_obj.group(1))

        if model_name not in results:
            results[model_name] = {}
        results[model_name][info_name] = {
            'miou': miou_value,
            'time': time_value
        }

    return results


def save_info(results, save_path):
    of = open(save_path, 'w')
    of.write("| 模型 | 使用TRT | 数值类型 | mIoU(%) | 耗时(ms/img) | FPS |\n")
    of.write("|  -  |   :-:   |   :-:  |  :-: |    :-:      |  :-:|\n")
    model_names = list(results.keys())
    model_names.sort()

    def write_helper(model_name, use_trt, dtype, info):
        miou = "-" if info["miou"] is None else round(100 * info["miou"], 2)
        time = "-" if info["time"] is None else info["time"]
        fps = "-" if info["time"] is None else round(1000.0 / info["time"], 2)
        of.write("| {:<60} | {} | {} | {:<6} | {:<8} | {:<6} | \n".format(
            model_name, use_trt, dtype, miou, time, fps))

    for model_name in model_names:
        info = results[model_name]
        if "Naive_fp32" in info:
            write_helper(model_name, "N", "FP32", info["Naive_fp32"])
        if "TRT_fp32" in info:
            write_helper(model_name, "Y", "FP32", info["TRT_fp32"])
        if "TRT_fp16" in info:
            write_helper(model_name, "Y", "FP16", info["TRT_fp16"])

    of.close()


if __name__ == '__main__':
    args = parse_args()
    results = analyze(args.log_path)
    save_info(results, args.save_path)

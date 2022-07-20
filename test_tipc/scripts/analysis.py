# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import print_function

import argparse
import json
import os
import re
import traceback
from numpy import mean, var


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filename", type=str, help="The name of log which need to analysis.")
    parser.add_argument("--speed_log_file", type=str, help="json file")
    parser.add_argument(
        "--log_with_profiler",
        type=str,
        help="The path of train log with profiler")
    parser.add_argument(
        "--profiler_path", type=str, help="The path of profiler timeline log.")
    parser.add_argument(
        "--keyword", type=str, help="Keyword to specify analysis data")
    parser.add_argument(
        "--separator",
        type=str,
        default=None,
        help="Separator of different field in log")
    parser.add_argument(
        '--position', type=int, default=None, help='The position of data field')
    parser.add_argument(
        '--range',
        type=str,
        default="",
        help='The range of data field to intercept')
    parser.add_argument(
        '--skip_steps',
        type=int,
        default=0,
        help='The number of steps to be skipped')
    parser.add_argument(
        '--model_mode',
        type=int,
        default=-1,
        help='Analysis mode, default value is -1')

    parser.add_argument(
        '--model_name',
        type=str,
        default="model_name",
        help='training model_name, transformer_base')
    parser.add_argument(
        '--base_batch_size', type=int, help='base_batch size on gpu')
    parser.add_argument('--fp_item', type=str, help='fp_item:fp16|fp32')
    parser.add_argument('--run_mode', type=str, default="DP", help='DP|MP|PP')
    parser.add_argument(
        '--convergence_key',
        type=str,
        default="",
        help="Keyword to specify loss data")
    parser.add_argument(
        '--speed_unit', type=str, default="images/s", help='IPS unit')
    parser.add_argument(
        '--device_num',
        type=str,
        default='N1C1',
        help='device_num:N1C1|N1C8|N4C32')
    args = parser.parse_args()
    args.separator = None if args.separator == "None" else args.separator
    return args


def _is_number(num):
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    result = pattern.match(num)
    if result:
        return True
    else:
        return False


class TimeAnalyzer(object):
    def __init__(self,
                 filename,
                 keyword=None,
                 separator=None,
                 position=None,
                 range="-1"):
        if filename is None:
            raise Exception("Please specify the filename!")

        if keyword is None:
            raise Exception("Please specify the keyword!")

        self.filename = filename
        self.keyword = keyword
        self.separator = separator
        self.position = position
        self.range = range
        self.records = None
        self._distil()

    def _distil(self):
        self.records = []
        with open(self.filename, "r") as f_object:
            lines = f_object.readlines()
            for line in lines:
                if self.keyword not in line:
                    continue
                try:
                    result = None

                    # Distil the string from a line.
                    line = line.strip()
                    line_words = line.split(
                        self.separator) if self.separator else line.split()
                    if args.position:
                        result = line_words[self.position]
                    else:
                        # Distil the string following the keyword.
                        for i in range(len(line_words) - 1):
                            if line_words[i] == self.keyword:
                                result = line_words[i + 1]
                                break

                    # Distil the result from the picked string.
                    if not self.range:
                        result = result[0:]
                    elif _is_number(self.range):
                        result = result[0:int(self.range)]
                    else:
                        result = result[int(self.range.split(":")[0]):int(
                            self.range.split(":")[1])]
                    self.records.append(float(result))
                except Exception as exc:
                    print("line is: {}; separator={}; position={}".format(
                        line, self.separator, self.position))

        print("Extract {} records: separator={}; position={}".format(
            len(self.records), self.separator, self.position))

    def _get_fps(self,
                 mode,
                 base_batch_size,
                 gpu_num,
                 avg_of_records,
                 unit=None):
        if mode == -1:
            assert unit, "Please set the unit when mode is -1."
            fps = gpu_num * avg_of_records
        elif mode == 0:
            # s/step -> samples/s
            fps = (base_batch_size * gpu_num) / avg_of_records
            unit = "samples/s"
        elif mode == 1:
            # steps/s -> steps/s
            fps = avg_of_records
            unit = "steps/s"
        elif mode == 2:
            # s/step -> steps/s
            fps = 1 / avg_of_records
            unit = "steps/s"
        elif mode == 3:
            # steps/s -> samples/s
            fps = base_batch_size * gpu_num * avg_of_records
            unit = "samples/s"
        elif mode == 4:
            # s/epoch -> s/epoch
            fps = avg_of_records
            unit = "s/epoch"
        else:
            ValueError("Unsupported analysis mode.")

        return fps, unit

    def analysis(self,
                 base_batch_size,
                 gpu_num=1,
                 skip_steps=0,
                 mode=-1,
                 unit=None):
        if base_batch_size <= 0:
            print("base_batch_size should larger than 0.")
            return 0, ''

        if len(self.records) <= (
                skip_steps + 10
        ):  # to address the condition which item of log equals to skip_steps
            print("ERROR!!! too few logs printed")
            return 0, ''

        sum_of_records = 0
        sum_of_records_skipped = 0
        skip_min = self.records[skip_steps]
        skip_max = self.records[skip_steps]

        count = len(self.records)
        # 1 计算skip 后平均值
        for i in range(count):
            sum_of_records += self.records[i]
            if i >= skip_steps:
                sum_of_records_skipped += self.records[i]
                if self.records[i] < skip_min:
                    skip_min = self.records[i]
                if self.records[i] > skip_max:
                    skip_max = self.records[i]
        avg_of_records = sum_of_records / float(count)
        avg_of_records_skipped = sum_of_records_skipped / float(count -
                                                                skip_steps)
        # 2 skip后去掉去除前max(5%,5)和后max(5%,5)个数据再计算平均值
        sorted_records = sorted(self.records[skip_steps:])
        skip_step2 = max(int(len(sorted_records) * 0.05), 5)
        try:
            del sorted_records[:skip_step2]
            del sorted_records[-skip_step2:]
            avg_of_sorted_records = mean(sorted_records)
            var_of_sorted_records = var(sorted_records)
            skip_min = min(sorted_records)
            skip_max = max(sorted_records)
        except Exception:
            print("no records")
            return 0, ''

        fps, fps_unit = self._get_fps(mode, base_batch_size, gpu_num,
                                      avg_of_records, unit)
        # fps_skipped, _ = self._get_fps(mode, base_batch_size, gpu_num, avg_of_records_skipped, unit)
        Fips, _ = self._get_fps(mode, base_batch_size, gpu_num,
                                avg_of_sorted_records, unit)

        if mode == -1:
            print("average ips of %d steps, skip 0 step:" % count)
            print("\tAvg: %.3f %s" % (avg_of_records, fps_unit))
            print("\tFPS: %.3f %s" % (fps, fps_unit))
            if skip_steps > 0:
                print("average ips of %d steps, skip %d steps, valid steps %d :" % (count, \
                    skip_steps, count - skip_steps - 2 * skip_step2))
                print("\tvar: %.3f " % (var_of_sorted_records))
                print("\tAvg: %.3f %s" % (avg_of_sorted_records, fps_unit))
                print("\tMin: %.3f %s" % (skip_min, fps_unit))
                print("\tMax: %.3f %s" % (skip_max, fps_unit))
                print("\tdiff_min_max: %.3f %s" % (
                    (skip_min - skip_max) * 100 / skip_max, "%"))
                print("\tFPS: %.3f %s" % (Fips, fps_unit))
        elif mode == 1 or mode == 3:
            print("average latency of %d steps, skip 0 step:" % count)
            print("\tAvg: %.3f steps/s" % avg_of_records)
            print("\tFPS: %.3f %s" % (fps, fps_unit))
            if skip_steps > 0:
                print("average latency of %d steps, skip %d steps:" %
                      (count, skip_steps))
                print("\tAvg: %.3f steps/s" % avg_of_records_skipped)
                print("\tMin: %.3f steps/s" % skip_min)
                print("\tMax: %.3f steps/s" % skip_max)
                print("\tFPS: %.3f %s" % (fps_skipped, fps_unit))
        elif mode == 0 or mode == 2:
            print("average latency of %d steps, skip 0 step:" % count)
            print("\tAvg: %.3f s/step" % avg_of_records)
            print("\tFPS: %.3f %s" % (fps, fps_unit))
            if skip_steps > 0:
                print("average latency of %d steps, skip %d steps:" %
                      (count, skip_steps))
                print("\tAvg: %.3f s/step" % avg_of_records_skipped)
                print("\tMin: %.3f s/step" % skip_min)
                print("\tMax: %.3f s/step" % skip_max)
                print("\tFPS: %.3f %s" % (fps_skipped, fps_unit))

        return round(Fips, 3), fps_unit


class ExceptionTest(Exception):
    pass


class LossAnalyzer(object):
    def __init__(self, filename, convergence_key=None, separator=None):
        if filename is None:
            raise Exception("Please specify the filename!")
        if convergence_key is None:
            raise Exception("Please specify the keyword of loss!")
        self.filename = filename
        self.convergence_key = convergence_key
        self.separator = separator

    def get_loss(self):
        with open(self.filename, "r") as f_object:
            lines = f_object.readlines()
            lines.reverse()
            result_loss = 0
            for line in lines:
                if self.convergence_key not in line:
                    continue
                try:
                    result_loss = 0
                    line = line.strip()
                    line_words = line.split(
                        self.separator) if self.separator else line.split()
                    for i in range(len(line_words) - 1):
                        if line_words[i] == self.convergence_key:
                            result_loss = line_words[i + 1]
                            result_loss = result_loss.replace(',', '')
                            raise ExceptionTest()
                except ExceptionTest:
                    break
        print("\tLoss: {}".format(result_loss))
        return result_loss


if __name__ == "__main__":
    args = parse_args()
    run_info = dict()
    run_info["model_branch"] = os.getenv("model_branch")
    run_info["model_commit"] = os.getenv("model_commit")
    run_info["model_name"] = args.model_name
    run_info["batch_size"] = args.base_batch_size
    run_info["fp_item"] = args.fp_item
    if re.match(
            r'DP.-MP.-PP.', args.run_mode
    ) or 'DP_MoE_C' in args.run_mode or 'Sharding_MoE_C' in args.run_mode:
        run_info["run_mode"] = 'Collective'
    else:
        run_info["run_mode"] = args.run_mode
    run_info["convergence_value"] = 0
    run_info["convergence_key"] = args.convergence_key
    run_info["ips"] = 0
    run_info["speed_unit"] = args.speed_unit
    run_info["device_num"] = args.device_num
    run_info["model_run_time"] = os.getenv('model_run_time')
    run_info["frame_commit"] = os.getenv('frame_commit')
    run_info["frame_version"] = os.getenv('frame_version')
    device_num = args.device_num
    print("---device_num:-", device_num)
    index_c = device_num.index('C')
    print("---index_c:-", index_c)
    gpu_num = int(device_num[index_c + 1:len(device_num)])
    print("-----gpu_num:", gpu_num)
    if "pwgan" in args.model_name:
        print("------analysis ", args.model_name)
        args.keyword = "avg_ips:"

    try:
        analyzer = TimeAnalyzer(args.filename, args.keyword, args.separator,
                                args.position, args.range)
        run_info["ips"], run_info["speed_unit"] = analyzer.analysis(
            base_batch_size=args.base_batch_size,
            gpu_num=gpu_num,
            skip_steps=args.skip_steps,
            mode=args.model_mode,
            unit=args.speed_unit)
        if args.convergence_key != "":
            loss_analyzer = LossAnalyzer(args.filename, args.convergence_key)
            run_info["convergence_value"] = loss_analyzer.get_loss()
    except Exception:
        traceback.print_exc()
    print("{}".format(json.dumps(run_info))
          )  # it's required, for the log file path  insert to the database
    with open(args.speed_log_file, "w") as f:
        f.write(json.dumps(run_info))

#!/usr/bin/env python

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from paddlepanseg.project_manager import work_on_project


def main(*args, **kwargs):
    meta_parser = argparse.ArgumentParser(add_help=False)
    meta_parser.add_argument('proj', type=str)
    meta_parser.add_argument(
        'cmd', type=str,
        choices=['train', 'val', 'predict', 'export', 'infer'])
    meta_args, rem = meta_parser.parse_known_args(*args, **kwargs)

    proj = meta_args.proj
    cmd = meta_args.cmd
    with work_on_project(proj):
        if cmd == 'train':
            import paddlepanseg.apis as api
            args = api.parse_train_args(rem)
            api.train_with_args(args)
        elif cmd == 'val':
            import paddlepanseg.apis as api
            args = api.parse_val_args(rem)
            api.val_with_args(args)
        elif cmd == 'predict':
            import paddlepanseg.apis as api
            args = api.parse_pred_args(rem)
            api.pred_with_args(args)
        elif cmd == 'export':
            import paddlepanseg.apis.deploy.export as export
            args = export.parse_export_args(rem)
            export.export_with_args(args)
        elif cmd == 'infer':
            import paddlepanseg.apis.deploy.python as infer
            args = infer.parse_infer_args(rem)
            infer.infer_with_args(args)


if __name__ == '__main__':
    main()

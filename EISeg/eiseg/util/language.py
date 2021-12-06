# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os.path as osp
import re
from eiseg import pjpath
from collections import defaultdict
import json
from urllib import parse
import requests


class TransUI(object):
    def __init__(self, is_trans=False):
        super().__init__()
        self.trans_dict = defaultdict(dict)
        with open(osp.join(pjpath, "config/zh_CN.EN"), "r", encoding="utf-8") as f:
            texts = f.readlines()
            for txt in texts:
                strs = txt.split("@")
                self.trans_dict[strs[0].strip()] = strs[1].strip()
        self.is_trans = is_trans
        self.youdao_url = "http://fanyi.youdao.com/translate?&doctype=json&type=AUTO&i="

    def put(self, zh_CN):
        if self.is_trans == False:
            return zh_CN
        else:
            try:
                return str(self.trans_dict[zh_CN])
            except:
                return zh_CN

    # 联网动态翻译
    def tr(self, zh_CN):
        try:
            tr_url = self.youdao_url + parse.quote(zh_CN)
            response = requests.get(tr_url)
            js = json.loads(response.text)
            result_EN = js["translateResult"][0][0]["tgt"]
            return str(result_EN)
        except:
            return zh_CN

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

import json
import random
import hashlib
from urllib import parse
import http.client
from tqdm import tqdm
from collections import defaultdict

from bs4 import BeautifulSoup as bs


class BaiduTranslate:
    def __init__(self, fromLang, toLang):
        self.url = "/api/trans/vip/translate"
        self.appid = "20200311000396156"
        self.secretKey = "s6c3ZeYTI9lhrwQVugnM"
        self.fromLang = fromLang
        self.toLang = toLang
        self.salt = random.randint(32768, 65536)

    def trans(self, text):
        sign = self.appid + text + str(self.salt) + self.secretKey
        md = hashlib.md5()
        md.update(sign.encode(encoding="utf-8"))
        sign = md.hexdigest()
        myurl = (self.url + "?appid=" + self.appid + "&q=" + parse.quote(text) +
                 "&from=" + self.fromLang + "&to=" + self.toLang + "&salt=" +
                 str(self.salt) + "&sign=" + sign)
        try:
            httpClient = http.client.HTTPConnection("api.fanyi.baidu.com")
            httpClient.request("GET", myurl)
            response = httpClient.getresponse()
            html = response.read().decode("utf-8")
            html = json.loads(html)
            dst = html["trans_result"][0]["dst"]
            return True, dst
        except Exception as e:
            return False, e


def read_ts(ts_path):
    xml = open(ts_path, "r", encoding="utf-8").read()
    xml = bs(xml, "xml")
    return xml


pre_ts_path = "tool/ts/English.ts"  # Russia
ts_path = "tool/ts/out.ts"
pre_xml = read_ts(pre_ts_path)
xml = read_ts(ts_path)
pre_messages = pre_xml.find_all("message")
messages = xml.find_all("message")
bd_trans = BaiduTranslate("auto", "en")  # ru
trans = bd_trans.trans

translated = 0
failed = 0
for msg in messages:
    type = msg.translation.get("type", None)
    source = msg.source.string
    trans = msg.translation.string
    if type == "unfinished" and trans is None and source is not None:
        in_pre = False
        for pmsg in pre_messages:
            if pmsg.source.string == source:
                try:
                    msg.translation.string = pmsg.translation.string
                    translated += 1
                    print(
                        f"{translated + failed} / {len(messages)}:{source} \t {msg.translation.string}"
                    )
                    in_pre = True
                except:
                    pass
                break
        if in_pre is False:
            res = bd_trans.trans(source)
            if res[0]:
                msg.translation.string = res[1]
                translated += 1
            else:
                failed += 1
            print(
                f"{translated + failed} / {len(messages)}:{source} \t {msg.translation.string}"
            )

for name in xml.find_all("name"):
    name.string = "APP_EISeg"

print(f"Totally {len(messages)} , translated {translated}, failed {failed}")
open(ts_path, "w", encoding="utf-8").write(str(xml))

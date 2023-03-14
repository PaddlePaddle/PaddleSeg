# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from tqdm import tqdm
import http.client
from bs4 import BeautifulSoup as bs


def read_ts(ts_path):
    xml = open(ts_path, "r", encoding="utf-8").read()
    xml = bs(xml, "xml")
    return xml


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


if __name__ == "__main__":
    bd_trans = BaiduTranslate("en", "ara")
    en_path = "tool/ts/English.ts"
    ar_path = "tool/ts/Arabic.ts"
    en_xml = read_ts(en_path)
    ar_xml = read_ts(ar_path)
    en_xml.find("TS")["language"] = "ar"
    en_mess = en_xml.find_all("message")
    ar_mess = ar_xml.find_all("message")
    for ems in tqdm(en_mess):
        need_trans = True
        for ams in ar_mess:
            if ems.source.string == ams.source.string and ams.translation.string is not None:
                ems.translation.string = ams.translation.string
                need_trans = False
                break
        if need_trans:
            pre_str = ""
            en_translate = ems.translation.string
            if "&" in en_translate:
                en_translate = en_translate.replace("&", "")
                pre_str = "&"
            res = bd_trans.trans(en_translate)
            if res[0]:
                ems.translation.string = pre_str + res[1]
            else:
                print("Can not translate: ", ems.source.string)
    open(
        ar_path.replace(".ts", "2.ts"), "w",
        encoding="utf-8").write(str(en_xml))

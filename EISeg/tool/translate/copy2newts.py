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

from tqdm import tqdm
from bs4 import BeautifulSoup as bs


def read_ts(ts_path):
    xml = open(ts_path, "r", encoding="utf-8").read()
    xml = bs(xml, "xml")
    return xml


if __name__ == "__main__":
    pre_path = "tool/ts/English.ts"
    new_path = "tool/ts/out.ts"
    pre_xml = read_ts(pre_path)
    new_xml = read_ts(new_path)
    pre_mess = pre_xml.find_all("message")
    new_mess = new_xml.find_all("message")
    for nms in tqdm(new_mess):
        new_source = nms.source.string
        type = nms.translation.get("type", None)
        for pms in pre_mess:
            pre_source = pms.source.string
            if new_source == pre_source and type == "unfinished":
                nms.translation.string = pms.translation.string
                del nms.translation["type"]
    open(
        pre_path.replace(".ts", "2.ts"), "w",
        encoding="utf-8").write(str(new_xml))

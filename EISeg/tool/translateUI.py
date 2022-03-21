import os
import os.path as osp
import json
import random
import hashlib
from urllib import parse
import http.client
from tqdm import tqdm
from collections import defaultdict


class BaiduTranslate:
    def __init__(self, fromLang, toLang):
        self.url = "/api/trans/vip/translate"
        self.appid = "20200311000396156"
        self.secretKey = "s6c3ZeYTI9lhrwQVugnM"
        self.fromLang = fromLang
        self.toLang = toLang
        self.salt = random.randint(32768, 65536)

    def BdTrans(self, text):
        sign = self.appid + text + str(self.salt) + self.secretKey
        md = hashlib.md5()
        md.update(sign.encode(encoding="utf-8"))
        sign = md.hexdigest()
        myurl = self.url + \
                "?appid=" + self.appid + \
                "&q=" + parse.quote(text) + \
                "&from=" + self.fromLang + \
                "&to=" + self.toLang + \
                "&salt=" + str(self.salt) + \
                "&sign=" + sign
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


# 获取所有可能带有ui的py文件
ui_files = []
widget_path = "eiseg/widget"
widget_names = os.listdir(widget_path)
for widget_name in widget_names:
    if widget_name != "__init__.py" and widget_name != "__pycache__":
        ui_files.append(osp.join(widget_path, widget_name))
ui_files.append("eiseg/ui.py")
ui_files.append("eiseg/app.py")

# 查找
chinese = []
keys = "trans.put(\""
for ui_file in ui_files:
    with open(ui_file, "r", encoding="utf-8") as f:
        codes = f.read()
        sp_codes = codes.split(keys)
        if len(sp_codes) == 1:
            continue
        else:
            sp_codes.pop(0)
            for sp_code in sp_codes:
                chinese.append(sp_code.split("\")")[0])
chinese = list(set(chinese))
# print(len(chinese))
# print(chinese)

# 比对（以前有的不重新机翻）
save_path = "eiseg/config/zh_CN.EN"
now_words = defaultdict(dict)
with open(save_path, "r", encoding="utf-8") as f:
    datas = f.readlines()
    for data in datas:
        words = data.strip().split("@")
        now_words[words[0]] = words[1]


# 翻译
def firstCharUpper(s):
    return s[:1].upper() + s[1:]


translate = []
baidu_trans = BaiduTranslate("zh", "en")
for cn in tqdm(chinese):
    if cn not in now_words.keys():
        en = baidu_trans.BdTrans(cn)
        tr = cn + "@" + firstCharUpper(en[-1])  # 首字母大写
    else:
        tr = cn + "@" + now_words[cn]
    translate.append(tr)

# 保存翻译内容
with open(save_path, "w", encoding="utf-8") as f:
    for language in translate:
        f.write(language + "\n")

print("trans OK!")

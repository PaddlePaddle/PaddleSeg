import re


# 检查中文
def check_cn(path):
    zh_model = re.compile(u'[\u4e00-\u9fa5]')
    return zh_model.search(path)


# 替换斜杠
def normcase(path):
    return eval(repr(path).replace('\\\\', '/'))  
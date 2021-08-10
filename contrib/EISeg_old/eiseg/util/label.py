import os
import os.path as osp


def _toint(seq):
    if isinstance(seq, list):
        for i in range(len(seq)):
            try:
                seq[i] = int(seq[i])
            except ValueError:
                pass
    else:
        seq = int(seq)
    return seq


def _saveLabel(labelList, path):
    if not path or len(path) == 0 or not osp.exists(osp.dirname(path)):
        return
    with open(path, "w", encoding="utf-8") as f:
        for ml in labelList:
            print(ml.idx, end=" ", file=f)
            print(ml.name, end=" ", file=f)
            for idx in range(3):
                print(ml.color[idx], end=" ", file=f)
            print(file=f)


def _readLabel(path):
    if not path or len(path) == 0 or not osp.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        labels = f.readlines()
    labelList = []
    for lab in labels:
        lab = lab.replace("\n", "").strip(" ").split(" ")
        if len(lab) != 2 and len(lab) != 5:
            continue
        label = _MaskLabel(_toint(lab[0]), str(lab[1]), _toint(lab[2:]))
        labelList.append(label)
    return labelList


class _MaskLabel:
    def __init__(self, idx=None, name=None, color=None):
        self.idx = idx
        self.name = name
        self.color = color


class Labeler(object):
    def __init__(self):
        self.list = []

    def add(self, idx, name, color):
        self.list.append(_MaskLabel(idx, name, color))

    def remove(self, index):
        del self.list[index]

    def clear(self):
        self.list = []

    def readLabel(self, path):
        self.list = _readLabel(path)

    def saveLabel(self, path):
        _saveLabel(self.list, path)

    def __getitem__(self, index):
        return self.list[index]

    def __len__(self):
        return len(self.list)

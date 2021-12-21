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

import os
import os.path as osp

from . import colorMap


class Label:
    def __init__(self, idx=None, name=None, color=None):
        self.idx = idx
        self.name = name
        self.color = color

    def __repr__(self):
        return f"{self.idx} {self.name} {self.color}"


class LabelList(object):
    def __init__(self, labels: dict = None):
        self.labelList = []
        if labels is not None:
            for lab in labels:
                color = lab.get("color", colorMap.get_color())
                self.add(lab["id"], lab["name"], color)

    def add(self, idx, name, color):
        self.labelList.append(Label(idx, name, color))

    def remove(self, index):
        for idx, lab in enumerate(self.labelList):
            if lab.idx == index:
                del self.labelList[idx]
                break
        # del self.labelList[index]

    def clear(self):
        self.labelList = []

    def toint(self, seq):
        if isinstance(seq, list):
            for i in range(len(seq)):
                try:
                    seq[i] = int(seq[i])
                except ValueError:
                    pass
        else:
            seq = int(seq)
        return seq

    def importLabel(self, path):
        if not osp.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            labels = f.readlines()
        labelList = []
        for lab in labels:
            lab = lab.replace("\n", "").strip(" ").split(" ")
            if len(lab) != 2 and len(lab) != 5:
                print(f"{lab} 标签不合法")
                continue
            label = Label(self.toint(lab[0]), str(lab[1]), self.toint(lab[2:]))
            labelList.append(label)
        self.labelList = labelList

    def exportLabel(self, path):
        if not path or not osp.exists(osp.dirname(path)):
            print("label path don't exist")
            return
        with open(path, "w", encoding="utf-8") as f:
            for label in self.labelList:
                print(label.idx, end=" ", file=f)
                print(label.name, end=" ", file=f)
                for idx in range(3):
                    print(label.color[idx], end=" ", file=f)
                print(file=f)

    def getLabelById(self, labelIdx):
        for lab in self.labelList:
            if lab.idx == labelIdx:
                return lab

    def __repr__(self):
        return str(self.labelList)

    def __getitem__(self, index):
        return self.labelList[index]

    def __len__(self):
        return len(self.labelList)

    @property
    def colors(self):
        cols = []
        for lab in self.labelList:
            cols.append(lab.color)
        return cols

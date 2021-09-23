import os.path as osp
import random

from eiseg import pjpath


class ColorMap(object):
    def __init__(self, color_path, shuffle=False):
        self.colors = []
        self.index = 0
        self.usedColors = []
        colors = open(color_path, "r").readlines()
        if shuffle:
            random.shuffle(colors)
        self.colors = [[int(x) for x in c.strip().split(",")] for c in colors]

    def get_color(self):
        color = self.colors[self.index]
        self.index = (self.index + 1) % len(self)
        return color

    def __len__(self):
        return len(self.colors)


colorMap = ColorMap(osp.join(pjpath, "config/colormap.txt"))

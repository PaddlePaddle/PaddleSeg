import os.path as osp
import random


class ColorMask(object):
    def __init__(
        self, color_path, shuffle=False
    ):
        self.color_maps = []
        self.index = 0
        with open(color_path, "r") as f:
            self.color_maps = f.readlines()
        if shuffle:
            random.shuffle(self.color_maps)
        self.color_map_nums = len(self.color_maps)

    def get_color(self):
        color = self.color_maps[self.index].strip()
        self.index += 1
        if self.index == self.color_map_nums:
            self.index = 0
        return self.to_list(color)

    def to_list(self, color):
        r, g, b = color.split(",")
        return [int(r), int(g), int(b)]
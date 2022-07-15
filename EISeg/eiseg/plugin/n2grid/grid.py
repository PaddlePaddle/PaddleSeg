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


import math
import numpy as np
from PIL import Image


def checkOpenGrid(img, thumbnail_min):
    H, W = img.shape[:2]
    if max(H, W) <= thumbnail_min:
        return False
    else:
        return True


class Grids:
    def __init__(self, img, gridSize=(512, 512), overlap=(24, 24)):
        self.clear()
        self.detimg = img
        self.gridSize = np.array(gridSize)
        self.overlap = np.array(overlap)

    def clear(self):
        # 图像HWC格式
        self.detimg = None  # 宫格初始图像
        self.grid_init = False  # 是否初始化了宫格
        # self.imagesGrid = []  # 图像宫格
        self.mask_grids = []  # 标签宫格
        self.grid_count = None  # (row count, col count)
        self.curr_idx = None  # (current row, current col)

    def createGrids(self):
        # 计算宫格横纵向格数
        imgSize = np.array(self.detimg.shape[:2])
        grid_count = np.ceil((imgSize + self.overlap) / self.gridSize)
        self.grid_count = grid_count = grid_count.astype("uint16")
        # ul = self.overlap - self.gridSize
        # for row in range(grid_count[0]):
        #     ul[0] = ul[0] + self.gridSize[0] - self.overlap[0]
        #     for col in range(grid_count[1]):
        #         ul[1] = ul[1] + self.gridSize[1] - self.overlap[1]
        #         lr = ul + self.gridSize
        #         # print("ul, lr", ul, lr)
        #         # 扩充
        #         det_tmp = self.detimg[ul[0]: lr[0], ul[1]: lr[1]]
        #         tmp = np.zeros((self.gridSize[0], self.gridSize[1], self.detimg.shape[-1]))
        #         tmp[:det_tmp.shape[0], :det_tmp.shape[1], :] = det_tmp
        #         self.imagesGrid.append(tmp)
        # self.mask_grids = [[np.zeros(self.gridSize)] * grid_count[1]] * grid_count[0]  # 不能用浅拷贝
        self.mask_grids = [
            [np.zeros(self.gridSize) for _ in range(grid_count[1])]
            for _ in range(grid_count[0])
        ]
        # print(len(self.mask_grids), len(self.mask_grids[0]))
        self.grid_init = True
        return list(grid_count)

    def getGrid(self, row, col):
        gridIdx = np.array([row, col])
        ul = gridIdx * (self.gridSize - self.overlap)
        lr = ul + self.gridSize
        img = self.detimg[ul[0]:lr[0], ul[1]:lr[1]]
        mask = self.mask_grids[row][col]
        self.curr_idx = (row, col)
        return img, mask

    def splicingList(self, save_path):
        """
        将slide的out进行拼接，raw_size保证恢复到原状
        """
        imgs = self.mask_grids
        raw_size = self.detimg.shape[:2]
        # h, w = None, None
        # for i in range(len(imgs)):
        #     for j in range(len(imgs[i])):
        #         im = imgs[i][j]
        #         if im is not None:
        #             h, w = im.shape[:2]
        #             break
        # if h is None and w is None:
        #     return False
        h, w = self.gridSize
        row = math.ceil(raw_size[0] / h)
        col = math.ceil(raw_size[1] / w)
        result_1 = np.zeros((h * row, w * col), dtype=np.uint8)
        result_2 = result_1.copy()
        # k = 0
        for i in range(row):
            for j in range(col):
                ih, iw = imgs[i][j].shape[:2]
                im = np.zeros(self.gridSize)
                im[:ih, :iw] = imgs[i][j]
                start_h = (i * h) if i == 0 else (i * (h - self.overlap[0]))
                end_h = start_h + h
                start_w = (j * w) if j == 0 else (j * (w - self.overlap[1]))
                end_w = start_w + w
                # 单区自己，重叠取或
                if (i + j) % 2 == 0:
                    result_1[start_h:end_h, start_w:end_w] = im
                else:
                    result_2[start_h:end_h, start_w:end_w] = im

        result = np.where(result_2 != 0, result_2, result_1)
        result = result[:raw_size[0], :raw_size[1]]
        Image.fromarray(result).save(save_path, "PNG")
        return result

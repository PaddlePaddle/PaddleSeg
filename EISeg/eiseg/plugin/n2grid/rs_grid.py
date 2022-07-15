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


import numpy as np
from typing import List, Tuple
from eiseg.plugin.remotesensing.raster import Raster


class RSGrids:
    def __init__(self, raset: Raster) -> None:
        """ 在EISeg中用于处理遥感栅格数据的宫格类.

        参数:
            tif_path (str): GTiff数据的路径.
            show_band (Union[List[int], Tuple[int]], optional): 用于RGB合成显示的波段. 默认为 [1, 1, 1].
            grid_size (Union[List[int], Tuple[int]], optional): 切片大小. 默认为 [512, 512].
            overlap (Union[List[int], Tuple[int]], optional): 重叠区域的大小. 默认为 [24, 24].
        """
        super(RSGrids, self).__init__()
        self.raster = raset
        self.clear()

    def clear(self) -> None:
        self.mask_grids = []  # 标签宫格
        self.grid_count = None  # (row count, col count)
        self.curr_idx = None  # (current row, current col)

    def createGrids(self) -> List[int]:
        img_size = (self.raster.geoinfo.ysize, self.raster.geoinfo.xsize)
        grid_count = np.ceil((img_size + self.raster.overlap) / self.raster.grid_size)
        self.grid_count = grid_count = grid_count.astype("uint16")
        self.mask_grids = [[np.zeros(self.raster.grid_size) \
                            for _ in range(grid_count[1])] for _ in range(grid_count[0])]
        return list(grid_count)

    def getGrid(self, row: int, col: int) -> Tuple[np.ndarray]:
        img, _ = self.raster.getGrid(row, col)
        mask = self.mask_grids[row][col]
        self.curr_idx = (row, col)
        return img, mask

    def splicingList(self, save_path: str) -> np.ndarray:
        mask = self.raster.saveMaskbyGrids(self.mask_grids, 
                                           save_path,
                                           self.raster.geoinfo)
        return mask
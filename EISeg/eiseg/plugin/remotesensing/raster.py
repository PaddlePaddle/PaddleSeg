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


import os.path as osp
import numpy as np
import cv2
import math
from typing import List, Dict, Tuple, Union
from collections import defaultdict
from easydict import EasyDict as edict
from .imgtools import sample_norm, two_percentLinear, get_thumbnail


def check_rasterio() -> bool:
    try:
        import rasterio
        return True
    except:
        return False


IMPORT_STATE = False
if check_rasterio():
    import rasterio
    from rasterio.windows import Window
    IMPORT_STATE = True


class Raster:
    def __init__(self, 
                 tif_path: str,
                 show_band: Union[List[int], Tuple[int]]=[1, 1, 1], 
                 open_grid: bool=False,
                 grid_size: Union[List[int], Tuple[int]]=[512, 512],
                 overlap: Union[List[int], Tuple[int]]=[24, 24]) -> None:
        """ 在EISeg中用于处理遥感栅格数据的类.

        参数:
            tif_path (str): GTiff数据的路径.
            show_band (Union[List[int], Tuple[int]], optional): 用于RGB合成显示的波段. 默认为 [1, 1, 1].
            open_grid (bool, optional): 是否打开了宫格切片功能. 默认为 False.
            grid_size (Union[List[int], Tuple[int]], optional): 切片大小. 默认为 [512, 512].
            overlap (Union[List[int], Tuple[int]], optional): 重叠区域的大小. 默认为 [24, 24].
        """
        super(Raster, self).__init__()
        if IMPORT_STATE is False:
            raise("Can't import rasterio!")
        if osp.exists(tif_path):
            self.src_data = rasterio.open(tif_path)
            self.geoinfo = self.__getRasterInfo()
            self.show_band = list(show_band)
            self.grid_size = np.array(grid_size)
            self.overlap = np.array(overlap)
            self.open_grid = open_grid
        else:
            raise("{0} not exists!".format(tif_path))
        self.thumbnail_min = 2000

    def __del__(self) -> None:
        self.src_data.close()

    def __getRasterInfo(self) -> Dict:
        meta = self.src_data.meta
        geoinfo = edict()
        geoinfo.count = meta["count"]
        geoinfo.dtype = meta["dtype"]
        geoinfo.xsize = meta["width"]
        geoinfo.ysize = meta["height"]
        geoinfo.geotf = meta["transform"]
        geoinfo.crs = meta["crs"]
        if geoinfo.crs is not None:
            geoinfo.crs_wkt = geoinfo.crs.wkt
        else:
            geoinfo.crs_wkt = None
        return geoinfo

    def checkOpenGrid(self, thumbnail_min: Union[int, None]) -> bool:
        if isinstance(thumbnail_min, int):
            self.thumbnail_min = thumbnail_min
        if max(self.geoinfo.xsize, self.geoinfo.ysize) <= self.thumbnail_min:
            self.open_grid = False
        else:
            self.open_grid = True
        return self.open_grid

    def setBand(self, bands: Union[List[int], Tuple[int]]) -> None:
        self.show_band = list(bands)

    # def __analysis_proj4(self) -> str:
    #     proj4 = self.geoinfo.crs.wkt  # TODO: 解析为proj4
    #     ap_dict = defaultdict(str)
    #     dinf = proj4.split("+")
    #     for df in dinf:
    #         kv = df.strip().split("=")
    #         if len(kv) == 2:
    #             k, v = kv
    #             ap_dict[k] = v
    #     return str("● 投影：{0}\n● 基准：{1}\n● 单位：{2}".format(
    #             ap_dict["proj"], ap_dict["datum"], ap_dict["units"])
    #     )

    def showGeoInfo(self) -> str:
        # return str("● 波段数：{0}\n● 数据类型：{1}\n● 行数：{2}\n● 列数：{3}\n{4}".format(
        #     self.geoinfo.count, self.geoinfo.dtype, self.geoinfo.xsize,
        #     self.geoinfo.ysize, self.__analysis_proj4())
        # )
        if self.geoinfo.crs is not None:
            crs = str(self.geoinfo.crs.to_string().split(":")[-1])
        else:
            crs = "None"
        return (str(self.geoinfo.count), str(self.geoinfo.dtype), str(self.geoinfo.xsize),
                str(self.geoinfo.ysize), crs)

    def getArray(self) -> Tuple[np.ndarray]:
        rgb = []
        if not self.open_grid:
            for b in self.show_band:
                rgb.append(self.src_data.read(b))
            geotf = self.geoinfo.geotf
        else:
            for b in self.show_band:
                rgb.append(get_thumbnail(self.src_data.read(b), self.thumbnail_min))
            geotf = None
        ima = np.stack(rgb, axis=2)  # cv2.merge(rgb)
        if self.geoinfo["dtype"] != "uint8":
            ima = sample_norm(ima)
        return two_percentLinear(ima), geotf

    def getGrid(self, row: int, col: int) -> Tuple[np.ndarray]:
        if self.open_grid is False:
            return self.getArray()
        grid_idx = np.array([row, col])
        ul = grid_idx * (self.grid_size - self.overlap)
        lr = ul + self.grid_size
        window = Window(ul[1], ul[0], (lr[1] - ul[1]), (lr[0] - ul[0]))
        rgb = []
        for b in self.show_band:
            rgb.append(self.src_data.read(b, window=window))
        win_tf = self.src_data.window_transform(window)
        ima = cv2.merge([np.uint16(c) for c in rgb])
        if self.geoinfo["dtype"] == "uint32":
            ima = sample_norm(ima)
        return two_percentLinear(ima), win_tf

    def saveMask(self, img: np.array, save_path: str, 
                 geoinfo: Union[Dict, None]=None, count: int=1) -> None:
        if geoinfo is None:
            geoinfo = self.geoinfo
        new_meta = self.src_data.meta.copy()
        new_meta.update({
            "driver": "GTiff",
            "width": geoinfo.xsize,
            "height": geoinfo.ysize,
            "count": count,
            "dtype": geoinfo.dtype,
            "crs": geoinfo.crs,
            "transform": geoinfo.geotf[:6],
            "nodata": 0
            })
        img = np.nan_to_num(img).astype("int16")
        with rasterio.open(save_path, "w", **new_meta) as tf:
            if count == 1:
                tf.write(img, indexes=1)
            else:
                for i in range(count):
                    tf.write(img[:, :, i], indexes=(i + 1))

    def saveMaskbyGrids(self, 
                        img_list: List[List[np.ndarray]], 
                        save_path: Union[str, None]=None,
                        geoinfo: Union[Dict, None]=None) -> np.ndarray:
        if geoinfo is None:
            geoinfo = self.geoinfo
        raw_size = (geoinfo.ysize, geoinfo.xsize)
        h, w = self.grid_size
        row = math.ceil(raw_size[0] / h)
        col = math.ceil(raw_size[1] / w)
        result_1 = np.zeros((h * row, w * col), dtype=np.uint8)
        result_2 = result_1.copy()
        for i in range(row):
            for j in range(col):
                ih, iw = img_list[i][j].shape[:2]
                im = np.zeros(self.grid_size)
                im[:ih, :iw] = img_list[i][j]
                start_h = (i * h) if i == 0 else (i * (h - self.overlap[0]))
                end_h = start_h + h
                start_w = (j * w) if j == 0 else (j * (w - self.overlap[1]))
                end_w = start_w + w
                # 单区自己，重叠取或
                if (i + j) % 2 == 0:
                    result_1[start_h: end_h, start_w: end_w] = im
                else:
                    result_2[start_h: end_h, start_w: end_w] = im
        result = np.where(result_2 != 0, result_2, result_1)
        result = result[:raw_size[0], :raw_size[1]]
        if save_path is not None:
            self.saveMask(result, save_path, geoinfo)
        return result
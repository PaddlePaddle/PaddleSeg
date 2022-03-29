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


def check_gdal():
    try:
        import gdal
    except:
        try:
            from osgeo import gdal
        except ImportError:
            return False
    return True


import numpy as np
import cv2
from collections import defaultdict

IPT_GDAL = check_gdal()
if IPT_GDAL:
    try:
        import gdal
        import osr
        import ogr
    except:
        from osgeo import gdal, osr, ogr


def open_tif(geoimg_path):
    """
    打开tif文件
    """
    if IPT_GDAL == True:
        geoimg = gdal.Open(geoimg_path)
        return __tif2arr(geoimg), get_geoinfo(geoimg)
    else:
        raise ImportError("can't import gdal!")


def __tif2arr(geoimg):
    if IPT_GDAL == True:
        tifarr = geoimg.ReadAsArray()
        if len(tifarr.shape) == 3:
            tifarr = tifarr.transpose((1, 2, 0))  # 多波段图像默认是[c, h, w]
        return tifarr
    else:
        raise ImportError("can't import gdal!")


def get_geoinfo(geoimg):
    """
    获取tif图像的信息，输入为dgal读取的数据
    """
    if IPT_GDAL == True:
        geoinfo = {
            "count": geoimg.RasterCount,
            "xsize": geoimg.RasterXSize,
            "ysize": geoimg.RasterYSize,
            "proj": geoimg.GetProjection(),
            "proj2": geoimg.GetSpatialRef(),
            "geotf": geoimg.GetGeoTransform(),
        }
        return geoinfo
    else:
        raise ImportError("can't import gdal!")


def show_geoinfo(geo_info, type):
    return str("● 波段数：{0}\n● 数据类型：{1}\n● 行数：{2}\n● 列数：{3}\n{4}".format(
        geo_info["count"],
        type,
        geo_info["xsize"],
        geo_info["ysize"],
        __analysis_proj2(geo_info["proj2"].ExportToProj4()), ))


def __analysis_proj2(proj2):
    ap_dict = defaultdict(str)
    dinf = proj2.split("+")
    for df in dinf:
        kv = df.strip().split("=")
        if len(kv) == 2:
            k, v = kv
            ap_dict[k] = v
    return str("● 投影：{0}\n● 基准：{1}\n● 单位：{2}".format(ap_dict["proj"], ap_dict[
        "datum"], ap_dict["units"]))


def save_tif(img, geoinfo, save_path):
    """
    保存分割的图像并使其空间信息保持一致
    """
    if IPT_GDAL == True:
        driver = gdal.GetDriverByName("GTiff")
        datatype = gdal.GDT_Byte
        dataset = driver.Create(
            save_path,
            geoinfo["xsize"],
            geoinfo["ysize"],
            1,  # geoinfo['count'],  # 保存tif目前仅用于保存mask，波段为1就OK
            datatype, )
        dataset.SetProjection(geoinfo["proj"])  # 写入投影
        dataset.SetGeoTransform(geoinfo["geotf"])  # 写入仿射变换参数
        # 同上
        # C = img.shape[-1] if len(img.shape) == 3 else 1
        # if C == 1:
        #     dataset.GetRasterBand(1).WriteArray(img)
        # else:
        #     for i_c in range(C):
        #         dataset.GetRasterBand(i_c + 1).WriteArray(img[:, :, i_c])
        dataset.GetRasterBand(1).WriteArray(img)
        del dataset  # 删除与tif的连接
    else:
        raise ImportError("can't import gdal!")

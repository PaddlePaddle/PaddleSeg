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
from collections import defaultdict
from typing import List, Dict


def check_gdal() -> bool:
    try:
        import gdal
    except:
        try:
            from osgeo import gdal
        except ImportError:
            return False
    return True


IMPORT_STATE = False
if check_gdal():
    try:
        import gdal
        import osr
        import ogr
    except:
        from osgeo import gdal, osr, ogr
    IMPORT_STATE = True


# 坐标系转换
def __convert_coord(point: List[float], g: List[float]) -> np.array:
    tform = np.array(g).reshape((3, 3))
    olp = np.ones((1, 3))
    olp[0, :2] = np.array(point)
    nwp = np.dot(tform, olp.T)
    return nwp.T[0, :2]


# 边界转为wkt格式
def __bound2wkt(bounds: List[Dict],
                tform: List[float],
                ct: osr.CoordinateTransformation) -> List[str]:
    geo_list = []
    for bd in bounds:
        gl = defaultdict()
        gl["clas"] = bd["name"]
        gl["polygon"] = "Polygon (("
        p = bd["points"]
        for i in range(len(p)):
            x, y = __convert_coord(p[i], tform)  # 仿射变换
            lon, lat = ct.TransformPoint(x, y)[:2]  # 转换到经纬度坐标
            gl["polygon"] += (str(lat) + " " + str(lon)) + ","
        x, y = __convert_coord(p[0], tform)  # 仿射变换
        lon, lat = ct.TransformPoint(x, y)[:2]  # 转换到经纬度坐标
        gl["polygon"] += (str(lat) + " " + str(lon)) + "))"
        # gl["polygon"] = gl["polygon"][:-1] + "))"
        geo_list.append(gl)
    return geo_list


# 保存shp文件
def save_shp(shp_path: str, geocode_list: List[Dict], geo_info: Dict) -> str:
    if IMPORT_STATE == True:
        # 支持中文路径
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
        # 属性表字段支持中文
        gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")
        # 注册驱动
        ogr.RegisterAll()
        # 创建shp数据
        strDriverName = "ESRI Shapefile"
        oDriver = ogr.GetDriverByName(strDriverName)
        if oDriver == None:
            return "驱动不可用：" + strDriverName
        # 创建数据源
        oDS = oDriver.CreateDataSource(shp_path)
        if oDS == None:
            return "创建文件失败：" + shp_path
        # 创建一个多边形图层
        prosrs = osr.SpatialReference()
        prosrs.ImportFromWkt(geo_info.crs_wkt)
        geosrs = prosrs.CloneGeogCS()
        ct = osr.CoordinateTransformation(prosrs, geosrs)
        geocode_list = __bound2wkt(geocode_list, geo_info.geotf, ct)
        ogr_type = ogr.wkbPolygon
        shpe_name = osp.splitext(osp.split(shp_path)[-1])[0]
        oLayer = oDS.CreateLayer(shpe_name, geosrs, ogr_type)
        if oLayer == None:
            return "图层创建失败！"
        # 创建属性表
        # 创建id字段
        oId = ogr.FieldDefn("id", ogr.OFTInteger)
        oLayer.CreateField(oId, 1)
        # 创建字段
        oAddress = ogr.FieldDefn("clas", ogr.OFTString)
        oLayer.CreateField(oAddress, 1)
        oDefn = oLayer.GetLayerDefn()
        # 创建要素
        # 数据集
        for index, f in enumerate(geocode_list):
            oFeaturePolygon = ogr.Feature(oDefn)
            oFeaturePolygon.SetField("id", index)
            oFeaturePolygon.SetField("clas", f["clas"])
            geomPolygon = ogr.CreateGeometryFromWkt(f["polygon"])
            oFeaturePolygon.SetGeometry(geomPolygon)
            oLayer.CreateFeature(oFeaturePolygon)
        # 创建完成后，关闭进程
        oDS.Destroy()
        return "数据集创建完成！"
    else:
        raise ImportError("can't import gdal, osr, ogr!")

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


# 保存shp文件
def save_shp(shp_path: str, tif_path: str, ignore_index :int=0) -> str:
    if IMPORT_STATE == True:
        ds = gdal.Open(tif_path)
        srcband = ds.GetRasterBand(1)
        maskband = srcband.GetMaskBand()
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
        gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")
        ogr.RegisterAll()
        drv = ogr.GetDriverByName("ESRI Shapefile")
        if osp.exists(shp_path):
            os.remove(shp_path)
        dst_ds = drv.CreateDataSource(shp_path)
        prosrs = osr.SpatialReference(wkt=ds.GetProjection())
        dst_layer = dst_ds.CreateLayer(
            "segmentation", geom_type=ogr.wkbPolygon, srs=prosrs)
        dst_fieldname = "DN"
        fd = ogr.FieldDefn(dst_fieldname, ogr.OFTInteger)
        dst_layer.CreateField(fd)
        gdal.Polygonize(srcband, maskband, dst_layer, 0, [])
        lyr = dst_ds.GetLayer()
        lyr.SetAttributeFilter("DN = '{}'".format(str(ignore_index)))
        for holes in lyr:
            lyr.DeleteFeature(holes.GetFID())
        dst_ds.Destroy()
        ds = None
        return "Dataset creation successfully!"
    else:
        raise ImportError("can't import gdal, osr, ogr!")
# Remote Sensing

This part presents documents related to remote sensing in EISeg, including its environment configuration and functions.

## 1 Environment Configuration

EISeg supports remote sensing data with GDAL and OGR. The former is a translator library for raster spatial data formats under the X/MIT style Open Source License, while the latter has similar functions but mainly supports vector data.

### 1.1 Install Dependencies

GDAL can be installed as follows:

#### 1.1.1 Windows

Windows users can download the corresponding  binaries (*.whl) of Python and system versions [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal). Here we take GDAL-3.3.3 -cp39-cp39-win_amd64.whl as an example, go to the download directory:

```
cd download
```

Install the dependencies:

```
pip install GDAL‑3.3.3‑cp39‑cp39‑win_amd64.whl
```

#### 1.1.2 Linux/Mac

Mac users are recommended to install with conda:

```
conda install gdal
```

## 2 Functions

At present, functions of remote sensing in EISeg are relatively simple including GTiff class data loading, large remote sensing image slicing and merging, and geographic raster/vector data (GTiff/ESRI Shapefile) export. What's more, an interactive model of building segmentation is trained based on more than 400,000 data from various building datasets.

### 2.1 Data Loading

For the moment, EISeg can only read remote sensing images with *.tif/tiff suffix. Since the training data are all remote sensing image slices of RGB three-channel, the interactive segmentation shares the same basis, which means EISeg supports band selection of multi-band data.

When adopting EISeg to open the GTiff image, the current number of bands is obtained, which can be set by the drop-down list of band settings. The default is [b1, b1, b1]. The following example shows the true color setting of Tiangong-1 multispectral data.

[![yd6fa-hqvvb](https://user-images.githubusercontent.com/71769312/141137443-a327309e-0987-4b2a-88fd-f698e08d3294.gif)](https://user-images.githubusercontent.com/71769312/141137443-a327309e-0987-4b2a-88fd-f698e08d3294.gif)

### 2.2 large Image Slicing

EISeg supports the post-prediction merging of sliced large remote sensing images (the latest attempt is 900M three-channel images with a size of 17000*10000), in which the overlap (overlapping area) of slices is 24.

[![140916007-86076366-62ce-49ba-b1d9-18239baafc90](https://user-images.githubusercontent.com/71769312/141139282-854dcb4f-bcab-4ccc-aa3c-577cc52ca385.png)](https://user-images.githubusercontent.com/71769312/141139282-854dcb4f-bcab-4ccc-aa3c-577cc52ca385.png)

The following demonstrates the slicing of some districts in Chongqing from Google Earth:

[![7kevx-q90hv](https://user-images.githubusercontent.com/71769312/141137447-60b305b1-a8ef-4b06-a45e-6db0b1ef2516.gif)](https://user-images.githubusercontent.com/71769312/141137447-60b305b1-a8ef-4b06-a45e-6db0b1ef2516.gif)

### 2.3 Geographic Data Saving

When the GTiff images to be labeled are accompanied by georeferencing, you can set EISeg to save them as GTiff with georeferencing or ESRI Shapefile.

- GTiff: A standard image file for industries of GIS and satellite remote sensing.
- ESRI Shapefile: The most common vector data format.The Shapefile file is a GIS file format developed by the U.S. Environmental Systems Research Institute (ESRI) and is the industry-standard vector data file. It is supported by all commercial and open source GIS software and now represents the industry standard.

[![82jlu-no59o](https://user-images.githubusercontent.com/71769312/141137726-76457454-5e9c-4ad0-85d6-d03f658ee63c.gif)](https://user-images.githubusercontent.com/71769312/141137726-76457454-5e9c-4ad0-85d6-d03f658ee63c.gif)

### 2.4 Labeling Model for Remote Sensing

[static_hrnet18_ocr48_rsbuilding_instance](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18_ocr48_rsbuilding_instance.zip) are recommended for building labeling.

English | [简体中文](tools.md)

# Tool scripting

There are some tool scriptings in EISeg in path of `EISeg/tool`.

## How to construct segmentation dataset for PaddleX

After completing image anotation by EISeg，by applying `eiseg2paddlex.py` in `tool` file, yoou can quickly convert data to PaddleX format for training. Execute the following command:

```
python eiseg2paddlex.py -d save_folder_path -o image_folder_path [-l label_folder_path] [-s split_rate]
```


- `save_folder_path`: path to save PaddleX format data.
- `image_folder_path`: path of data to be converted.
- `label_folder_path`:  path of the label, it is not required, if it is not filled, default is "image_folder_path/label".
- `split_rate`: The devision ratio of training set and validation set, default is 0.9.

![68747470733a2f2f73332e626d702e6f76682f696d67732f323032312f31302f373134633433396139633766613439622e706e67](https://user-images.githubusercontent.com/71769312/141392744-f1a27774-2714-43a2-8808-2fc14a5a6b5a.png)

## Semantic labels to instance labels

The semantic segmentation label is converted to the instance segmentation label (the original label is in range \[0,255\], and the result is a single-channel image that uses a palette to color. Through the `semantic2instance.py`, the semantic segmentation data marked by EISeg can be converted into instance segmentation data. Use the following method:

``` shell
python semantic2instance.py -o label_path -d save_path
```

Parameters:

- `label_path`: path to semantic label, required.
- `save_path`: path to instance label, required.

![68747470733a2f2f73332e626d702e6f76682f696d67732f323032312f30392f303038633562373638623765343737612e706e67](https://user-images.githubusercontent.com/71769312/141392781-d99ec177-f445-4336-9ab2-0ba7ae75d664.png)

## Video Cut

Due to the huge amount of video data calculation, in order to prevent insufficient video memory, it is recommended to segment the video into 100 frames and label it again. You can find this script in `EISeg/tool/cut_video.py`

## Convert medical slice into video

3D medical annotation is based on video annotation algorithm. Therefore, before medical image annotation, medical image needs to be converted to `mp4` format and then labeled. You can find this script in `EISeg/tool/medical2video.py`

## Convert json of labelme to COCO

EISeg will generate a `labelme` folder under the `label` folder after the label is saved, which contains json files and `labels. txt` files with the same format as labelme. At this time, if you want to convert json files into coco files with the same format as labelme, you can execute the following command:

```shell
python labelme-json2labelme-coco.py label_path save_path --labels txt_path.

# E.g.
# python labelme-json2labelme-coco.py mydata/label/labelme/ mydata/label/labelme/output --labels mydata/label/labelme/labels.txt
```

Parameters:

- `label_path`: path to json format of label, required.
- `save_path`: path to save coco format data, required.
- `txt_path`: path to labels.txt, required.

After conversion, it will be displayed in `save_ Path` generates three files, namely `annotations. json`, `JPEGImages` and `Visualization`, in the same format as labelme.
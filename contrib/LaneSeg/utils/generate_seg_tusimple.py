import json
import numpy as np
import cv2
import os
import argparse
# this code is from https://raw.githubusercontent.com/ZJULearning/resa/main/tools/generate_seg_tusimple.py

TRAIN_SET = ['label_data_0313.json', 'label_data_0601.json']
VAL_SET = ['label_data_0531.json']
TRAIN_VAL_SET = TRAIN_SET + VAL_SET
TEST_SET = ['test_label.json']


def gen_label_for_json(args, image_set):
    H, W = 720, 1280
    SEG_WIDTH = 30
    save_dir = args.savedir

    os.makedirs(os.path.join(args.root, args.savedir, "list"), exist_ok=True)
    list_f = open(
        os.path.join(args.root, args.savedir, "list",
                     "{}_gt.txt".format(image_set)), "w")

    json_path = os.path.join(args.root, args.savedir,
                             "{}.json".format(image_set))
    with open(json_path) as f:
        for line in f:
            label = json.loads(line)
            # ---------- clean and sort lanes -------------
            lanes = []
            _lanes = []
            slope = [
            ]  # identify 0th, 1st, 2nd, 3rd, 4th, 5th lane through slope
            for i in range(len(label['lanes'])):
                l = [(x, y)
                     for x, y in zip(label['lanes'][i], label['h_samples'])
                     if x >= 0]
                if (len(l) > 1):
                    _lanes.append(l)
                    slope.append(
                        np.arctan2(l[-1][1] - l[0][1], l[0][0] - l[-1][0]) /
                        np.pi * 180)
            _lanes = [_lanes[i] for i in np.argsort(slope)]
            slope = [slope[i] for i in np.argsort(slope)]

            idx = [None for i in range(6)]
            for i in range(len(slope)):
                if slope[i] <= 90:
                    idx[2] = i
                    idx[1] = i - 1 if i > 0 else None
                    idx[0] = i - 2 if i > 1 else None
                else:
                    idx[3] = i
                    idx[4] = i + 1 if i + 1 < len(slope) else None
                    idx[5] = i + 2 if i + 2 < len(slope) else None
                    break
            for i in range(6):
                lanes.append([] if idx[i] is None else _lanes[idx[i]])

            # ---------------------------------------------

            img_path = label['raw_file']
            seg_img = np.zeros((H, W, 3))
            list_str = []  # str to be written to list.txt
            for i in range(len(lanes)):
                coords = lanes[i]
                if len(coords) < 4:
                    continue
                for j in range(len(coords) - 1):
                    cv2.line(seg_img, coords[j], coords[j + 1],
                             (i + 1, i + 1, i + 1), SEG_WIDTH // 2)

            seg_path = img_path.split("/")
            seg_path, img_name = os.path.join(
                args.root, args.savedir, seg_path[1], seg_path[2]), seg_path[3]
            os.makedirs(seg_path, exist_ok=True)
            seg_path = os.path.join(seg_path, img_name[:-3] + "png")
            cv2.imwrite(seg_path, seg_img)

            seg_path = "/".join([
                args.savedir, *img_path.split("/")[1:3], img_name[:-3] + "png"
            ])
            if seg_path[0] != '/':
                seg_path = '/' + seg_path
            if img_path[0] != '/':
                img_path = '/' + img_path
            list_str.insert(0, seg_path)
            list_str.insert(0, img_path)
            list_str = " ".join(list_str) + "\n"
            list_f.write(list_str)


def generate_json_file(save_dir, json_file, image_set):
    with open(os.path.join(save_dir, json_file), "w") as outfile:
        for json_name in (image_set):
            with open(os.path.join(args.root, json_name)) as infile:
                for line in infile:
                    outfile.write(line)


def generate_label(args):
    save_dir = os.path.join(args.root, args.savedir)
    os.makedirs(save_dir, exist_ok=True)
    generate_json_file(save_dir, "train_val.json", TRAIN_VAL_SET)
    generate_json_file(save_dir, "test.json", TEST_SET)

    print("generating train_val set...")
    gen_label_for_json(args, 'train_val')
    print("generating test set...")
    gen_label_for_json(args, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root', required=True, help='The root of the Tusimple dataset')
    parser.add_argument(
        '--savedir',
        type=str,
        default='seg_label',
        help='The root of the Tusimple dataset')
    args = parser.parse_args()

    generate_label(args)

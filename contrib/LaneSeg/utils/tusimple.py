import paddle.nn as nn
import json
import os
import cv2
import numpy as np

from .lane import LaneEval

# this code heavily base on
# https://github.com/ZJULearning/resa/blob/main/runner/evaluator/tusimple/tusimple.py
# https://github.com/ZJULearning/resa/blob/main/datasets/tusimple.py

def split_path(path):
    """split path tree into list"""
    folders = []
    while True:
        path, folder = os.path.split(path)
        if folder != "":
            folders.insert(0, folder)
        else:
            if path != "":
                folders.insert(0, path)
            break
    return folders


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


class Tusimple:
    def __init__(self,
                 num_classes=2,
                 cut_height=0,
                 thresh=0.6,
                 is_show=False,
                 test_gt_json=None,
                 save_dir='output/eval'):
        super(Tusimple, self).__init__()
        self.num_classes = num_classes
        self.cut_height = cut_height
        self.dump_to_json = []
        self.thresh = thresh  # probability threshold

        self.save_dir = save_dir
        self.is_show = is_show
        self.test_gt_json = test_gt_json
        self.smooth = True  # whether to smooth the probability or not
        self.y_px_gap = 10  # y pixel gap for sampling
        self.pts = 56  # y pixel gap for sampling
        self.target_shape = (720, 1280)
        self.color_map = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                          (255, 0, 255), (0, 255, 125), (50, 100, 50),
                          (100, 50, 100)]

    def evaluate(self, output, im_path):
        seg_pred = output[0]
        seg_pred = nn.functional.softmax(seg_pred, axis=1)
        seg_pred = seg_pred.numpy()
        self.generate_files(seg_pred, im_path)

    def predict(self, output, im_path):
        seg_pred = output[0]
        seg_pred = nn.functional.softmax(seg_pred, axis=1)
        seg_pred = seg_pred.numpy()
        img_path = im_path
        lane_coords_list = self.get_lane_coords(seg_pred)

        for batch in range(len(seg_pred)):
            lane_coords = lane_coords_list[batch]
            if True:
                img = cv2.imread(img_path)
                im_file = os.path.basename(im_path)
                saved_path = os.path.join(self.save_dir, 'points', im_file)
                self.draw(img, lane_coords, saved_path)

    def calculate_eval(self):
        output_file = os.path.join(self.save_dir, 'predict_test.json')
        with open(output_file, "w+") as f:
            for line in self.dump_to_json:
                print(line, end="\n", file=f)

        eval_rst, acc, fp, fn = LaneEval.bench_one_submit(
            output_file, self.test_gt_json)
        self.dump_to_json = []
        return acc, fp, fn, eval_rst

    def draw(self, img, coords, file_path=None):
        for i, coord in enumerate(coords):
            for x, y in coord:
                if x <= 0 or y <= 0:
                    continue
                x, y = int(x), int(y)
                cv2.circle(img, (x, y), 4, self.color_map[i % self.num_classes],
                           2)

        if file_path is not None:
            mkdir(file_path)
            cv2.imwrite(file_path, img)

    def generate_files(self, seg_pred, im_path):
        img_path = im_path
        lane_coords_list = self.get_lane_coords(seg_pred)

        for batch in range(len(seg_pred)):
            lane_coords = lane_coords_list[batch]
            json_dict = {}
            json_dict['lanes'] = []
            json_dict['h_sample'] = []
            json_dict['raw_file'] = os.path.join(
                *split_path(img_path[batch])[-4:])
            json_dict['run_time'] = 0
            for l in lane_coords:
                if len(l) == 0:
                    continue
                json_dict['lanes'].append([])
                for (x, y) in l:
                    json_dict['lanes'][-1].append(int(x))
            for (x, y) in lane_coords[0]:
                json_dict['h_sample'].append(y)
            self.dump_to_json.append(json.dumps(json_dict))
            if self.is_show:
                img = cv2.imread(img_path[batch])
                new_img_name = '_'.join(
                    [x for x in split_path(img_path[batch])[-4:]])

                saved_path = os.path.join(self.save_dir, 'vis', new_img_name)
                self.draw(img, lane_coords, saved_path)

    def get_lane_coords(self, seg_pred):
        lane_coords_list = []
        for batch in range(len(seg_pred)):
            seg = seg_pred[batch]
            lane_coords = self.heatmap2lane(seg, self.target_shape)
            for i in range(len(lane_coords)):
                lane_coords[i] = sorted(
                    lane_coords[i], key=lambda pair: pair[1])
            lane_coords_list.append(lane_coords)
        return lane_coords_list

    def process_gap(self, coordinate):
        if any(x > 0 for x in coordinate):
            start = [i for i, x in enumerate(coordinate) if x > 0][0]
            end = [
                i for i, x in reversed(list(enumerate(coordinate))) if x > 0
            ][0]
            lane = coordinate[start:end + 1]
            # The line segment is not continuous
            if any(x < 0 for x in lane):
                gap_start = [
                    i for i, x in enumerate(lane[:-1])
                    if x > 0 and lane[i + 1] < 0
                ]
                gap_end = [
                    i + 1 for i, x in enumerate(lane[:-1])
                    if x < 0 and lane[i + 1] > 0
                ]
                gap_id = [i for i, x in enumerate(lane) if x < 0]
                if len(gap_start) == 0 or len(gap_end) == 0:
                    return coordinate
                for id in gap_id:
                    for i in range(len(gap_start)):
                        if i >= len(gap_end):
                            return coordinate
                        if id > gap_start[i] and id < gap_end[i]:
                            gap_width = float(gap_end[i] - gap_start[i])
                            # line interpolation
                            lane[id] = int((id - gap_start[i]) / gap_width *
                                           lane[gap_end[i]] +
                                           (gap_end[i] - id) / gap_width *
                                           lane[gap_start[i]])
                if not all(x > 0 for x in lane):
                    print("Gaps still exist!")
                coordinate[start:end + 1] = lane
        return coordinate

    def is_short(self, lane):
        start = [i for i, x in enumerate(lane) if x > 0]
        if not start:
            return 1
        else:
            return 0

    def get_lane(self, prob_map, target_shape=None):
        """
        Arguments:
        ----------
        prob_map: prob map for single lane, np array size (h, w)
        target_shape:  reshape size target, (H, W)

        Return:
        ----------
        coords: x coords bottom up every y_px_gap px, 0 for non-exist, in resized shape
        """
        if target_shape is None:
            target_shape = prob_map.shape
        h, w = prob_map.shape
        H, W = target_shape
        H -= self.cut_height

        coords = np.zeros(self.pts)
        coords[:] = -1.0
        for i in range(self.pts):
            y = int((H - 10 - i * self.y_px_gap) * h / H)
            if y < 0:
                break
            line = prob_map[y, :]
            id = np.argmax(line)
            val = line[id]
            if val > self.thresh:
                coords[i] = int(id / w * W)
        if (coords > 0).sum() < 2:
            coords = np.zeros(self.pts)
        self.process_gap(coords)
        self.fix_outliers(coords)
        self.process_gap(coords)
        return coords

    def fix_outliers(self, coords):
        data = [x for i, x in enumerate(coords) if x > 0]
        index = [i for i, x in enumerate(coords) if x > 0]
        if len(data) == 0:
            return coords
        diff = []
        is_outlier = False
        n = 1
        x_gap = abs((data[-1] - data[0]) / (1.0 * (len(data) - 1)))
        for idx, dt in enumerate(data):
            if is_outlier == False:
                t = idx - 1
                n = 1
            if idx == 0:
                diff.append(0)
            else:
                diff.append(abs(data[idx] - data[t]))
                if abs(data[idx] - data[t]) > n * (x_gap * 1.5):
                    n = n + 1
                    is_outlier = True
                    ind = index[idx]
                    coords[ind] = -1
                else:
                    is_outlier = False

    def heatmap2lane(self, seg_pred, target_shape=(720, 1280)):
        """
        Arguments:
        ----------
        seg_pred:      np.array size (5, h, w)
        target_shape:  reshape size target, (H, W)

        Return:
        ----------
        coordinates: [x, y] list of lanes, e.g.: [ [[9, 569], [50, 549]] ,[[630, 569], [647, 549]] ]
        """
        if target_shape is None:
            target_shape = seg_pred.shape[1:]  # seg_pred (5, h, w)
        _, h, w = seg_pred.shape
        H, W = target_shape
        coordinates = []

        for i in range(self.num_classes - 1):
            prob_map = seg_pred[i + 1]
            if self.smooth:
                prob_map = cv2.blur(
                    prob_map, (9, 9), borderType=cv2.BORDER_REPLICATE)
            coords = self.get_lane(prob_map, target_shape)
            if self.is_short(coords):
                continue
            self.add_coords(coordinates, coords, H)

        if len(coordinates) == 0:
            coords = np.zeros(self.pts)
            self.add_coords(coordinates, coords, H)
        return coordinates

    def add_coords(self, coordinates, coords, H):
        sub_lanes = []
        for j in range(self.pts):
            if coords[j] > 0:
                val = [coords[j], H - 10 - j * self.y_px_gap]
            else:
                val = [-1, H - 10 - j * self.y_px_gap]
            sub_lanes.append(val)
        coordinates.append(sub_lanes)

import os
import re
import time
import random
import sys
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
import cv2
import skimage.io
from skimage.draw import polygon
from skimage import measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from rasterio import features

import solaris as sol
from shapely.ops import cascaded_union
from shapely.geometry import shape, Polygon


def get_respond_img(npy_file):
    sp = npy_file.split('.')[0].split('_')
    aoi = '_'.join(sp[5:])
    src_img_path = os.path.join(src_img_root, aoi, "images_masked",
                                npy_file.replace(".npy", ".tif"))
    return src_img_path


def get_building_polygon(mask,
                         thres_h,
                         thres_l,
                         distance,
                         min_area,
                         polygon_buffer,
                         conn=2,
                         watershed_line=True):
    mask0 = mask > thres_h
    local_maxi = peak_local_max(
        mask,
        indices=False,
        footprint=np.ones((distance * 2 + 1, distance * 2 + 1)),
        labels=(mask > thres_l))
    local_maxi[mask0] = True
    seed_msk = ndi.label(local_maxi)[0]

    mask = watershed(
        -mask, seed_msk, mask=(mask > thres_l), watershed_line=watershed_line)
    mask = measure.label(mask, connectivity=conn, background=0).astype('uint8')

    geoms_np = []
    geoms_polygons = []
    polygon_generator = features.shapes(mask, mask)
    for polygon, value in polygon_generator:
        p = shape(polygon)
        if polygon_buffer:
            p = p.buffer(polygon_buffer)
        if p.area >= min_area:
            p = p.simplify(tolerance=0.5)
            geoms_polygons.append(p)
            try:
                p = np.array(p.boundary.xy, dtype='int32').T
            except:
                p = np.array(p.boundary[0].xy, dtype='int32').T
            geoms_np.append(p)
    return geoms_np, geoms_polygons


def get_idx_in_polygon(contour):
    contour = contour.reshape(-1, 2) - 0.5
    p = Polygon(contour).buffer(-0.25)
    r, c = p.boundary.xy
    rr, cc = polygon(r, c)
    return rr, cc


def save_as_csv(file_path, npy_list, contours, footprints, npys):
    aoi = '_'.join(npy_list[0].split('/')[-1].split('.')[0].split('_')[5:])
    print('save csv: %s, npoly = %d' % (aoi, footprints.sum()))
    fw = open(file_path[:-4] + '_' + aoi + '.csv', 'w')
    for j, contour in enumerate(contours):
        contour = contour / 3
        p = Polygon(contour)
        p = p.simplify(tolerance=0.2)
        try:
            contour = np.array(p.boundary.xy, dtype='float32').T
        except:
            contour = np.array(p.boundary[0].xy, dtype='float32').T
        contour = np.round(contour.reshape(-1, 2) * 10) / 10

        polygon_str = re.sub(r"[\[\]]", '', ",".join(map(str, contour)))
        polygon_str = polygon_str.replace(". ", ' ')
        polygon_str = polygon_str.replace(".,", ',')
        polygon_str = re.sub(r" {2,}", ' ', polygon_str)
        polygon_str = re.sub(r" {0,}, {0,}", ',', polygon_str)

        for i, npy_file in enumerate(npy_list):
            filename = npy_file.split('/')[-1].split('.')[0]
            flag = footprints[j, i]
            if flag:
                fw.write(
                    "%s,%d,\"POLYGON ((%s))\"\n" % (filename, j, polygon_str))
    fw.close()


def check_ious(poly1,
               area1,
               center1,
               poly2s,
               area2s,
               center2s,
               thr,
               center_thr,
               eps=1e-5):
    dis = np.linalg.norm(center2s - center1, axis=1)
    r = (np.sqrt(area2s) + np.sqrt(area1)) / np.pi
    locs = np.where(dis < center_thr * r)[0]
    for idx in locs:
        poly2 = poly2s[idx]
        intersection = poly1.intersection(poly2).area
        if intersection < eps:
            continue
        else:
            union = poly1.union(poly2).area
            iou_score = intersection / float(union)
            if iou_score >= thr:
                return False
    return True


def filter_countours(polygons, filter_polygons, margin):
    filter_valid_polygons = []
    filter_areas = []
    filter_centers = []
    for p in filter_polygons:
        if margin:
            p = p.buffer(np.sqrt(p.area) * margin)
        if p.is_valid:
            filter_valid_polygons.append(p)
            filter_areas.append(p.area)
            filter_centers.append(np.array(p.bounds).reshape(2, 2).sum(1))
    filter_areas = np.array(filter_areas)
    filter_centers = np.array(filter_centers)

    filter_res = np.zeros(len(polygons), 'bool')
    for idx, p in enumerate(polygons):
        if p.is_valid:
            area = p.area
            center = np.array(p.bounds).reshape(2, 2).sum(1)
            if check_ious(
                    p,
                    area,
                    center,
                    filter_valid_polygons,
                    filter_areas,
                    filter_centers,
                    thr=0,
                    center_thr=5):
                filter_res[idx] = True
    filter_idxs = np.where(filter_res)[0]
    return filter_idxs


def process(npy_list,
            thres1=0.1,
            thres2_h1=0.6,
            thres2_l1=0.4,
            thres2_h2=0.6,
            thres2_l2=0.35,
            thres3_1=0.3,
            thres3_s=0.45,
            thres3_d=0.5,
            thres3_i=0,
            thres3_m=0.4,
            margin=0,
            distance=5,
            min_area=25.5,
            polygon_buffer=0):
    npy_list = sorted(npy_list)
    npy_sum = 0
    ignore_sum = 0
    npys = []
    for iii, npy_file in enumerate(npy_list):
        npy = np.load(npy_file)
        img_file = get_respond_img(npy_file.split('/')[-1])
        src_img = skimage.io.imread(img_file)
        mask = src_img[:, :, 3] == 0
        mask = np.repeat(mask, 3, axis=0)
        mask = np.repeat(mask, 3, axis=1)
        assert mask.shape[0] == src_img.shape[0] * 3 and mask.shape[
            1] == src_img.shape[1] * 3

        npy = npy[:mask.shape[0], :mask.shape[1]]
        npy[mask] = -10000
        npys.append(npy)

        ignore_mask = (npy > thres1)
        npy_sum = npy_sum + npy * ignore_mask
        ignore_sum = ignore_sum + ignore_mask

    npy_mean = npy_sum / np.maximum(ignore_sum, 1)

    npys = np.array(npys)
    img_num = npys.shape[0]
    ''' ============================ For Change Detection ============================ '''
    contours, polygons = get_building_polygon(
        npy_mean,
        thres2_h1,
        thres2_l1,
        distance=distance,
        min_area=min_area,
        polygon_buffer=polygon_buffer)

    building_num = len(contours)

    score_map = np.zeros((building_num, img_num))

    footprints = np.zeros_like(score_map)

    changeids = []
    for i, contour in enumerate(contours):
        rr, cc = get_idx_in_polygon(contour)
        point_filter = (rr < npys.shape[2]) & (cc < npys.shape[1]) & (
            cc >= 0) & (rr >= 0)
        rr = rr[point_filter]
        cc = cc[point_filter]

        scores = np.mean(npys[:, cc, rr], axis=1)
        score_mask = scores >= 0

        score_filter = np.zeros(img_num, dtype='bool')
        max_score = scores.max()

        masked_scores = scores[score_mask]
        left_mean = np.cumsum(masked_scores) / (
            np.arange(len(masked_scores)) + 1)
        right_mean = (np.cumsum(masked_scores[::-1]) /
                      (np.arange(len(masked_scores)) + 1))[::-1]

        max_diff = 0
        for idx in range(len(masked_scores) - 1):
            diff = right_mean[idx + 1] - left_mean[idx]
            max_diff = max(diff, max_diff)
            if max_diff > thres3_d:
                break
        if max_diff > thres3_d:
            changeids.append(i)
            start = False
            for idx, score in enumerate(scores):
                if not start:
                    if idx == 0 and score > thres3_1:
                        score_filter[idx] = 1
                        start = True
                    if score > thres3_s * max_score:
                        score_filter[idx] = 1
                        start = True
                else:
                    if score > thres3_i:
                        score_filter[idx] = 1

        footprints[i] = score_filter

    changeids = np.array(changeids)
    change_contours = [contours[idx] for idx in changeids]
    change_polygons = [polygons[idx] for idx in changeids]
    change_footprints = footprints[changeids]

    # print('num change footprints:', len(changeids), change_footprints.sum())
    ''' ============================ For Tracking ============================ '''
    contours, polygons = get_building_polygon(
        npy_mean,
        thres2_h2,
        thres2_l2,
        distance=distance,
        min_area=min_area,
        polygon_buffer=polygon_buffer)
    # print('num track footprints (before filter):', len(contours))
    filter_idx = filter_countours(polygons, change_polygons, margin)
    contours = [contours[idx] for idx in filter_idx]
    # print('num track footprints (after filter):', len(contours))

    building_num = len(contours)
    score_map = np.zeros((building_num, img_num))

    footprints = np.zeros_like(score_map)
    for i, contour in enumerate(contours):
        rr, cc = get_idx_in_polygon(contour)
        point_filter = (rr < npys.shape[2]) & (cc < npys.shape[1]) & (
            cc >= 0) & (rr >= 0)
        rr = rr[point_filter]
        cc = cc[point_filter]

        scores = np.mean(npys[:, cc, rr], axis=1)
        score_mask = scores >= 0

        score_filter = np.zeros(img_num, dtype='bool')
        max_score = scores.max()

        masked_scores = scores[score_mask]
        left_mean = np.cumsum(masked_scores) / (
            np.arange(len(masked_scores)) + 1)
        right_mean = (np.cumsum(masked_scores[::-1]) /
                      (np.arange(len(masked_scores)) + 1))[::-1]

        if scores[scores >= 0].mean() > thres3_m:
            score_filter[scores >= 0] = 1

        footprints[i] = score_filter

    final_contours = change_contours + contours
    final_footprints = np.concatenate([change_footprints, footprints], 0)
    save_as_csv(out_file, npy_list, final_contours, final_footprints, npys)


def main():
    dic = {}
    npy_files = [os.path.join(pred_root, x) for x in os.listdir(pred_root)]
    for npy_file in npy_files:
        key = '_'.join(npy_file.split('/')[-1].split('.')[0].split('_')[5:])
        if key not in dic:
            dic[key] = [npy_file]
        else:
            dic[key].append(npy_file)

    if os.path.isfile(out_file):
        os.remove(out_file)
    with open(out_file, 'w') as fw:
        fw.write("filename,id,geometry\n")

    params = []
    for aoi, npy_list in dic.items():
        print("Process:", aoi)
        params.append(npy_list)

    print("Execute!")
    print("len params:", len(params))
    n_threads = 10
    pool = multiprocessing.Pool(n_threads)
    _ = pool.map(process, params)

    for aoi in dic.keys():
        print('Merge:', aoi)
        with open(out_file, 'a') as fw:
            with open(out_file[:-4] + '_' + aoi + '.csv', 'r') as fr:
                for line in fr.readlines():
                    fw.write(line.strip() + '\n')
    for aoi in dic.keys():
        try:
            os.remove(out_file[:-4] + '_' + aoi + '.csv')
        except:
            pass
    print("Finish!")


if __name__ == "__main__":
    src_img_root = sys.argv[1]
    pred_root = sys.argv[2]
    out_file = sys.argv[3]
    main()

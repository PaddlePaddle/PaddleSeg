# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is based on https://github.com/PaddlePaddle/PaddleRS
Ths copyright of PaddlePaddle/PaddleRS is as follows:
Apache License [see LICENSE for details]
"""

import math

import cv2
import numpy as np

from .utils import calc_distance

S = 20
TD = 3
D = TD + 1

ALPHA = math.degrees(math.pi / 6)
BETA = math.degrees(math.pi * 17 / 18)
DELTA = math.degrees(math.pi / 12)
THETA = math.degrees(math.pi / 4)


def boundary_regularization(contour, mask_shape, W: int=32) -> np.ndarray:
    new_contour = _coarse(contour, mask_shape)  # coarse
    if new_contour is not None:
        contour = _fine(new_contour, W)  # fine
    return contour


def _coarse(contour, img_shape):
    def _inline_check(point, shape, eps=5):
        x, y = point[0]
        iH, iW = shape
        if x < eps or x > iH - eps or y < eps or y > iW - eps:
            return False
        else:
            return True

    area = cv2.contourArea(contour)
    # S = 20
    if area < S:  # remove polygons whose area is below a threshold S
        return None
    # D = 0.3 if area < 200 else 1.0
    # TD = 0.5 if area < 200 else 0.9
    epsilon = 0.005 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)  # DP
    p_number = contour.shape[0]
    idx = 0
    while idx < p_number:
        last_point = contour[idx - 1]
        current_point = contour[idx]
        next_idx = (idx + 1) % p_number
        next_point = contour[next_idx]
        # remove edges whose lengths are below a given side length TD
        # that varies with the area of a building.
        distance = calc_distance(current_point, next_point)
        if distance < TD and not _inline_check(next_point, img_shape):
            contour = np.delete(contour, next_idx, axis=0)
            p_number -= 1
            continue
        # remove over-sharp angles with threshold α.
        # remove over-smooth angles with threshold β.
        angle = _calc_angle(last_point, current_point, next_point)
        if (ALPHA > angle or angle > BETA) and _inline_check(current_point,
                                                             img_shape):
            contour = np.delete(contour, idx, axis=0)
            p_number -= 1
            continue
        idx += 1
    if p_number > 2:
        return contour
    else:
        return None


def _fine(contour, W):
    # area = cv2.contourArea(contour)
    # W = 6 if area < 200 else 8
    # TD = 0.5 if area < 200 else 0.9
    # D = TD + 0.3
    nW = W
    p_number = contour.shape[0]
    distance_list = []
    azimuth_list = []
    indexs_list = []
    for idx in range(p_number):
        current_point = contour[idx]
        next_idx = (idx + 1) % p_number
        next_point = contour[next_idx]
        distance_list.append(calc_distance(current_point, next_point))
        azimuth_list.append(_calc_azimuth(current_point, next_point))
        indexs_list.append((idx, next_idx))
    # add the direction of the longest edge to the list of main direction.
    longest_distance_idx = np.argmax(distance_list)
    main_direction_list = [azimuth_list[longest_distance_idx]]
    max_dis = distance_list[longest_distance_idx]
    if max_dis <= nW:
        nW = max_dis - 1e-6
    # Add other edges’ direction to the list of main directions
    # according to the angle threshold δ between their directions
    # and directions in the list.
    for distance, azimuth in zip(distance_list, azimuth_list):
        for mdir in main_direction_list:
            abs_dif_ang = abs(mdir - azimuth)
            if distance > nW and THETA <= abs_dif_ang <= (180 - THETA):
                main_direction_list.append(azimuth)
    contour_by_lines = []
    md_used_list = [main_direction_list[0]]
    for distance, azimuth, (idx, next_idx) in zip(distance_list, azimuth_list,
                                                  indexs_list):
        p1 = contour[idx]
        p2 = contour[next_idx]
        pm = (p1 + p2) / 2
        # find long edges with threshold W that varies with building’s area.
        if distance > nW:
            rotate_ang = main_direction_list[0] - azimuth
            for main_direction in main_direction_list:
                r_ang = main_direction - azimuth
                if abs(r_ang) < abs(rotate_ang):
                    rotate_ang = r_ang
                    md_used_list.append(main_direction)
            abs_rotate_ang = abs(rotate_ang)
            # adjust long edges according to the list and angles.
            if abs_rotate_ang < DELTA or abs_rotate_ang > (180 - DELTA):
                rp1 = _rotation(p1, pm, rotate_ang)
                rp2 = _rotation(p2, pm, rotate_ang)
            elif (90 - DELTA) < abs_rotate_ang < (90 + DELTA):
                rp1 = _rotation(p1, pm, rotate_ang - 90)
                rp2 = _rotation(p2, pm, rotate_ang - 90)
            else:
                rp1, rp2 = p1, p2
        # adjust short edges (judged by a threshold θ) according to the list and angles.
        else:
            rotate_ang = md_used_list[-1] - azimuth
            abs_rotate_ang = abs(rotate_ang)
            if abs_rotate_ang < THETA or abs_rotate_ang > (180 - THETA):
                rp1 = _rotation(p1, pm, rotate_ang)
                rp2 = _rotation(p2, pm, rotate_ang)
            else:
                rp1 = _rotation(p1, pm, rotate_ang - 90)
                rp2 = _rotation(p2, pm, rotate_ang - 90)
        # contour_by_lines.extend([rp1, rp2])
        contour_by_lines.append([rp1[0], rp2[0]])
    correct_points = np.array(contour_by_lines)
    # merge (or connect) parallel lines if the distance between
    # two lines is less than (or larger than) a threshold D.
    final_points = []
    final_points.append(correct_points[0][0].reshape([1, 2]))
    lp_number = correct_points.shape[0] - 1
    for idx in range(lp_number):
        next_idx = (idx + 1) if idx < lp_number else 0
        cur_edge_p1 = correct_points[idx][0]
        cur_edge_p2 = correct_points[idx][1]
        next_edge_p1 = correct_points[next_idx][0]
        next_edge_p2 = correct_points[next_idx][1]
        L1 = _line(cur_edge_p1, cur_edge_p2)
        L2 = _line(next_edge_p1, next_edge_p2)
        A1 = _calc_azimuth([cur_edge_p1], [cur_edge_p2])
        A2 = _calc_azimuth([next_edge_p1], [next_edge_p2])
        dif_azi = abs(A1 - A2)
        # find intersection point if not parallel
        if (90 - DELTA) < dif_azi < (90 + DELTA):
            point_intersection = _intersection(L1, L2)
            if point_intersection is not None:
                final_points.append(point_intersection)
        # move or add lines when parallel
        elif dif_azi < 1e-6:
            marg = _calc_distance_between_lines(L1, L2)
            if marg < D:
                # move
                point_move = _calc_project_in_line(next_edge_p1, cur_edge_p1,
                                                   cur_edge_p2)
                final_points.append(point_move)
                # update next
                correct_points[next_idx][0] = point_move
                correct_points[next_idx][1] = _calc_project_in_line(
                    next_edge_p2, cur_edge_p1, cur_edge_p2)
            else:
                # add line
                add_mid_point = (cur_edge_p2 + next_edge_p1) / 2
                rp1 = _calc_project_in_line(add_mid_point, cur_edge_p1,
                                            cur_edge_p2)
                rp2 = _calc_project_in_line(add_mid_point, next_edge_p1,
                                            next_edge_p2)
                final_points.extend([rp1, rp2])
        else:
            final_points.extend(
                [cur_edge_p1[np.newaxis, :], cur_edge_p2[np.newaxis, :]])
    final_points = np.array(final_points)
    return final_points


def _get_priority(hierarchy):
    if hierarchy[3] < 0:
        return 1
    if hierarchy[2] < 0:
        return 2
    return 3


def _fill(img, coarse_conts):
    result = np.zeros_like(img)
    sorted(coarse_conts, key=lambda x: x[1])
    for contour, priority in coarse_conts:
        if priority == 2:
            cv2.fillPoly(result, [contour.astype(np.int32)], (0, 0, 0))
        else:
            cv2.fillPoly(result, [contour.astype(np.int32)], (255, 255, 255))
    return result


def _calc_angle(p1, vertex, p2):
    x1, y1 = p1[0]
    xv, yv = vertex[0]
    x2, y2 = p2[0]
    a = ((xv - x2) * (xv - x2) + (yv - y2) * (yv - y2))**0.5
    b = ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))**0.5
    c = ((x1 - xv) * (x1 - xv) + (y1 - yv) * (y1 - yv))**0.5
    return math.degrees(math.acos((b**2 - a**2 - c**2) / (-2 * a * c)))


def _calc_azimuth(p1, p2):
    x1, y1 = p1[0]
    x2, y2 = p2[0]
    if y1 == y2:
        return 0.0
    if x1 == x2:
        return 90.0
    elif x1 < x2:
        if y1 < y2:
            ang = math.atan((y2 - y1) / (x2 - x1))
            return math.degrees(ang)
        else:
            ang = math.atan((y1 - y2) / (x2 - x1))
            return 180 - math.degrees(ang)
    else:  # x1 > x2
        if y1 < y2:
            ang = math.atan((y2 - y1) / (x1 - x2))
            return 180 - math.degrees(ang)
        else:
            ang = math.atan((y1 - y2) / (x1 - x2))
            return math.degrees(ang)


def _rotation(point, center, angle):
    if angle == 0:
        return point
    x, y = point[0]
    cx, cy = center[0]
    radian = math.radians(abs(angle))
    if angle > 0:  # clockwise
        rx = (x - cx) * math.cos(radian) - (y - cy) * math.sin(radian) + cx
        ry = (x - cx) * math.sin(radian) + (y - cy) * math.cos(radian) + cy
    else:
        rx = (x - cx) * math.cos(radian) + (y - cy) * math.sin(radian) + cx
        ry = (y - cy) * math.cos(radian) - (x - cx) * math.sin(radian) + cy
    return np.array([[rx, ry]])


def _line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def _intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return np.array([[x, y]])
    else:
        return None


def _calc_distance_between_lines(L1, L2):
    eps = 1e-16
    A1, _, C1 = L1
    A2, B2, C2 = L2
    new_C1 = C1 / (A1 + eps)
    new_A2 = 1
    new_B2 = B2 / (A2 + eps)
    new_C2 = C2 / (A2 + eps)
    dist = (np.abs(new_C1 - new_C2)) / (
        np.sqrt(new_A2 * new_A2 + new_B2 * new_B2) + eps)
    return dist


def _calc_project_in_line(point, line_point1, line_point2):
    eps = 1e-16
    m, n = point
    x1, y1 = line_point1
    x2, y2 = line_point2
    F = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
    x = (m * (x2 - x1) * (x2 - x1) + n * (y2 - y1) * (x2 - x1) +
         (x1 * y2 - x2 * y1) * (y2 - y1)) / (F + eps)
    y = (m * (x2 - x1) * (y2 - y1) + n * (y2 - y1) * (y2 - y1) +
         (x2 * y1 - x1 * y2) * (x2 - x1)) / (F + eps)
    return np.array([[x, y]])

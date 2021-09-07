from enum import Enum

import cv2
from math import sqrt
import matplotlib.pyplot as plt


class Instructions(Enum):
    No_Instruction = 0
    Polygon_Instruction = 1


def get_polygon(label, sample="Dynamic"):
    '''
        sample(int/float/str): 简化系数，设置为"Dynamic"表示根据面积简化，不可设置其他的str
    '''
    results = cv2.findContours(
        image=label, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_TC89_KCOS
    )  # 获取内外边界，用RETR_TREE更好表示
    cv2_v = cv2.__version__.split(".")[0]
    contours = results[1] if cv2_v == "3" else results[0]  # 边界
    hierarchys = results[2] if cv2_v == "3" else results[1]  # 隶属信息
    if len(contours) != 0:  # 可能出现没有边界的情况
        polygons = []
        relas = []
        for contour, hierarchy in zip(contours, hierarchys[0]):
            epsilon = 0.0005 * cv2.arcLength(contour, True) if sample == "Dynamic" else sample
            if not isinstance(epsilon, float) and not isinstance(epsilon, int):
                epsilon = 0
            out = cv2.approxPolyDP(contour, epsilon, True)
            # 判断自己，如果是子对象就不管自己是谁
            if hierarchy[2] == -1:
                own = None
            else:
                if hierarchy[0] == -1 and hierarchy[1] == -1:
                    own = 0
                elif hierarchy[0] != -1 and hierarchy[1] == -1:
                    own = hierarchy[0] - 1
                else:
                    own = hierarchy[1] + 1
            rela = (own,  # own
                    hierarchy[-1] if hierarchy[-1] != -1 else None)  # parent
            polygon = []
            for p in out:
                polygon.append(p[0])
            polygons.append(polygon)  # 边界
            relas.append(rela)  # 关系
        for i in range(len(relas)):
            if relas[i][1] != None:  # 有父母
                for j in range(len(relas)):
                    if relas[j][0] == relas[i][1]:  # i的父母就是j（i是j的内圈）
                        min_i, min_o = _find_min_point(polygons[i], polygons[j])
                        # 改变顺序
                        s_pj = polygons[j][: min_o]
                        polygons[j] = polygons[j][min_o:]
                        polygons[j].extend(s_pj)
                        s_pi = polygons[i][: min_i]
                        polygons[i] = polygons[i][min_i:]
                        polygons[i].extend(s_pi)
                        # 连接
                        polygons[j].append(polygons[j][0])  # 闭合
                        polygons[j].extend(polygons[i])
                        polygons[j].append(polygons[i][0])  # 闭合
                        polygons[i] = None
        polygons = list(filter(None, polygons))  # 清除加到外圈的内圈多边形
        return polygons
    else:
        print("没有标签范围，无法生成边界")
        return None


def _find_min_point(i_list, o_list):
    min_dis = 1e7
    idx_i = -1
    idx_o = -1
    for i in range(len(i_list)):
        for o in range(len(o_list)):
            dis = sqrt((i_list[i][0] - o_list[o][0]) ** 2 + \
                       (i_list[i][1] - o_list[o][1]) ** 2)
            if dis < min_dis:
                min_dis = dis
                idx_i = i
                idx_o = o
    return idx_i, idx_o
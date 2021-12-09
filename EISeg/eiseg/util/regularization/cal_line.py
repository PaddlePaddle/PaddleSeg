import numpy as np


# 线生成函数
def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


# 计算两条直线之间的交点
def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


# 计算两个平行线之间的距离
def par_line_dist(L1, L2):
    A1, B1, C1 = L1
    A2, B2, C2 = L2
    new_A1 = 1
    new_B1 = B1 / A1
    new_C1 = C1 / A1
    new_A2 = 1
    new_B2 = B2 / A2
    new_C2 = C2 / A2
    dist = (np.abs(new_C1 - new_C2)) / (np.sqrt(new_A2**2 + new_B2**2))
    return dist


# 计算点在直线的投影位置
def point_in_line(m, n, x1, y1, x2, y2):
    x = (m * (x2 - x1) * (x2 - x1) + n * (y2 - y1) * (x2 - x1) + 
        (x1 * y2 - x2 * y1) * (y2 - y1)) / ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
    y = (m * (x2 - x1) * (y2 - y1) + n * (y2 - y1) * (y2 - y1) + 
        (x2 * y1 - x1 * y2) * (x2 - x1)) / ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
    return (x, y)




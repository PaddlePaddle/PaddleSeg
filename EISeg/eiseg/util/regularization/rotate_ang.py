import math


# 顺时针旋转
def Nrotation_angle_get_coor_coordinates(point, center, angle):
    src_x, src_y = point
    center_x, center_y = center
    radian = math.radians(angle)
    dest_x = (src_x - center_x) * math.cos(radian) + \
             (src_y - center_y) * math.sin(radian) + center_x
    dest_y = (src_y - center_y) * math.cos(radian) - \
             (src_x - center_x) * math.sin(radian) + center_y
    # return (int(dest_x), int(dest_y))
    return (dest_x, dest_y)


# 逆时针旋转
def Srotation_angle_get_coor_coordinates(point, center, angle):
    src_x, src_y = point
    center_x, center_y = center
    radian = math.radians(angle)
    dest_x = (src_x - center_x) * math.cos(radian) - \
             (src_y - center_y) * math.sin(radian) + center_x
    dest_y = (src_x - center_x) * math.sin(radian) + \
             (src_y - center_y) * math.cos(radian) + center_y
    # return [int(dest_x), int(dest_y)]
    return (dest_x, dest_y)
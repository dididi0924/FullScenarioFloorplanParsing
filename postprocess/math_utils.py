import math
import numpy as np
from shapely.geometry import Point, LineString
from copy import deepcopy
import torch


def cal_length(pt1, pt2):
    """计算两点间距离"""

    vector1 = np.array(pt1)
    vector2 = np.array(pt2)
    
    return np.linalg.norm(vector1-vector2)

def cal_pty_in_line(pt1, pt2, x):
    if pt2[0] - pt1[0] == 0:
        y = pt1[1]
    else:
        y = (x - pt1[0]) / (pt2[0] - pt1[0]) * (pt2[1] - pt1[1]) + pt1[1]

    return y

def cal_ptx_in_line(pt1, pt2, y):
    if pt2[1] - pt1[1] == 0:
        x = pt1[0]
    else:
        x = (y - pt1[1]) / (pt2[1] - pt1[1]) * (pt2[0] - pt1[0]) + pt1[0]

    return x

def line_intersection(line1, line2):
    """计算两条直线交点"""
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    div = det(xdiff, ydiff)

    # 不相交
    if div == 0:
       return False

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return round(x), round(y)

def is_line(pt1, pt2, pt3):
    """判断三个点是否在一条直线上"""
    
    a = pt2[0] - pt1[0]
    b = pt2[1] - pt1[1]
    e = pt3[0] - pt1[0]
    f = pt3[1] - pt1[1]
 
    return a*f == e*b

def is_contained_pt(line, pt):
    """判断点是否在线段上, 宽松"""
    line = np.array(line)
    return min(line[..., 0]) <= pt[0] <= max(line[..., 0]) and min(line[..., 1]) <= pt[1] <= max(line[..., 1])

def is_contained_pt2(line, pt):
    """判断点是否在线段上, 严格"""
    line = LineString(line)
    point = Point(pt)

    return point.within(line)

def get_foot(line, pt):
    """计算垂足"""
    start_x, start_y = line[0][0], line[0][1]
    end_x, end_y = line[1][0], line[1][1]
    pa_x, pa_y = pt
 
    p_foot = [0, 0]
    if start_x == end_x:
        p_foot[0] = start_x
        p_foot[1] = pa_y
        return p_foot
    k = (end_y - start_y) * 1.0 / (end_x - start_x)
    a = k
    b = -1.0
    c = start_y - k * start_x
    p_foot[0] = (b * b * pa_x - a * b * pa_y - a * c) / (a * a + b * b)
    p_foot[1] = (a * a * pa_y - a * b * pa_x - b * c) / (a * a + b * b)

    return p_foot

def polygonToRotRectangle(bbox):
    """
    :param bbox: The polygon stored in format [x1, y1, x2, y2, x3, y3, x4, y4]
    :return: Rotated Rectangle in format [cx, cy, w, h, theta]
    """
    bbox = np.array(bbox, dtype=np.float32)
    bbox = np.reshape(bbox, newshape=(2, 4), order='F')
    angle = math.atan2(-(bbox[0, 1]-bbox[0, 0]), bbox[1, 1]-bbox[1, 0])

    center = [[0], [0]]

    for i in range(4):
        center[0] += bbox[0, i]
        center[1] += bbox[1, i]

    center = np.array(center, dtype=np.float32)/4.0

    R = np.array([[math.cos(angle), -math.sin(angle)],
                  [math.sin(angle), math.cos(angle)]], dtype=np.float32)

    normalized = np.matmul(R.transpose(), bbox-center)

    xmin = np.min(normalized[0, :])
    xmax = np.max(normalized[0, :])
    ymin = np.min(normalized[1, :])
    ymax = np.max(normalized[1, :])

    w = xmax - xmin + 1
    h = ymax - ymin + 1

    return [float(center[0]), float(center[1]), w, h, angle]

def get_max_inner_rect(contour):
    pts = contour.reshape(contour.shape[0], 2)
    # pts为轮廓坐标
    # 列表中存储元素分别为左上角，右上角，右下角和左下角
    rect = np.zeros((4, 2), dtype = "float32")
    # 左上角的点具有最小的和，而右下角的点具有最大的和
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 计算点之间的差值
    # 右上角的点具有最小的差值,
    # 左下角的点具有最大的差值
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # 返回排序坐标(依次为左上右上右下左下)
    return rect.flatten()

def needRevise(pt1, pt2, line):
    if cal_length(pt1, line[0]) > cal_length(pt1, line[0]):
        line[0], line[1] = line[1], line[0]
    if (pt1 == line[0]).all() or (pt2 == line[1]).all():
        return False
    res = False
    mid_pt = [(pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2]
    pt1_foot = get_foot(line, pt1)
    pt2_foot = get_foot(line, pt2)
    mid_foot = get_foot(line, mid_pt)
    try:
        if cal_angle([pt1, pt2], line) < 30 and (is_contained_pt(line, pt1_foot) or is_contained_pt(line, pt2_foot)) and cal_length(mid_pt, mid_foot) < 15:
            if projection_is_overlap([pt1, pt2], line):
                res = False
    except:
        res = False

    return res

def cal_angle(v1, v2):
    """求向量夹角"""
    x1 = v1[0][0] - v1[1][0]
    y1 = v1[0][1] - v1[1][1]

    x2 = v2[0][0] - v2[1][0]
    y2 = v2[0][1] - v2[1][1]
    try:
        angle = math.degrees(math.acos((x1 * x2 + y1 * y2) / (((x1 ** 2 + y1 ** 2) ** 0.5) * ((x2 ** 2 + y2 ** 2) ** 0.5))))
    except:
        angle = math.degrees(math.acos((x1 * x2 + y1 * y2) / (((x1 ** 2 + y1 ** 2) ** 0.5) * ((x2 ** 2 + y2 ** 2) ** 0.5) + 0.0001)))

    return 180 - angle % 180

def cal_angle_2(v1, v2):
    """求向量夹角"""
    x1 = v1[0][0] - v1[1][0]
    y1 = v1[0][1] - v1[1][1]

    x2 = v2[0][0] - v2[1][0]
    y2 = v2[0][1] - v2[1][1]
    try:
        angle = math.degrees(math.acos((x1 * x2 + y1 * y2) / (((x1 ** 2 + y1 ** 2) ** 0.5) * ((x2 ** 2 + y2 ** 2) ** 0.5))))
    except:
        angle = math.degrees(math.acos((x1 * x2 + y1 * y2) / (((x1 ** 2 + y1 ** 2) ** 0.5) * ((x2 ** 2 + y2 ** 2) ** 0.5) + 0.0001)))

    return angle

def cal_pt_line_segment_dis(pt, line):
    """点到线段的最短距离"""

    foot = get_foot(line, pt)

    if is_contained_pt(line, foot):
        return cal_length(foot, pt)
    else:
        return min(cal_length(pt, line[0]), cal_length(pt, line[1]))

def cal_pt_line_dis(pt, line):
    """点到直线的距离"""

    foot = get_foot(line, pt)
    return cal_length(pt, foot)

def cal_pt_line_dis_2(pt, line):
    """点到直线的距离"""

    p = Point(pt[0], pt[1])
    line = LineString(line)

    return p.distance(line)

def projection_is_overlap(line1, line2, overlap=10):
    """投影是否重合"""
    
    if cal_length(line1[0], line2[0]) > cal_length(line1[0], line2[1]):
        line2[0], line2[1] = line2[1], line2[0]
    flag = False
    # print(line1, line2)
    if (line1[1] == line2[0]).all() or (line1[0] == line2[1]).all():
        flag = False

    elif is_contained_pt(line2, get_foot(line2, line1[0])):
        # print("00")
        # 包含
        if is_contained_pt(line2, get_foot(line2, line1[1])):
            flag = True
        elif is_contained_pt(line1, get_foot(line1, line2[1])):
            # print("1")
            if cal_length(get_foot(line2, line1[0]), line2[1]) > overlap:
                flag = True
        elif is_contained_pt(line1, get_foot(line1, line2[0])):
            # print("2", cal_length(get_foot(line2, line1[0]), line2[0]), overlap)
            if cal_length(get_foot(line2, line1[0]), line2[0]) > overlap:
                flag = True
        
    elif is_contained_pt(line1, get_foot(line1, line2[0])):
        # print("11")
        # 包含于直线内
        if is_contained_pt(line1, get_foot(line1, line2[1])):
            flag = True
        elif is_contained_pt(line2, get_foot(line2, line1[1])):
            # print("3")
            if cal_length(get_foot(line1, line2[0]), line1[1]) > overlap:
                flag = True
        elif is_contained_pt(line2, get_foot(line2, line1[0])):
            # print("4")
            if cal_length(get_foot(line1, line2[0]), line1[0]) > overlap:
                flag = True
        
    return flag

def projection_len(a, b):
    """line1 向 line2 做投影，计算投影的长度"""
    line1 = deepcopy(a)
    line2 = deepcopy(b)
    overlap = 0
    if cal_length(line1[0], line2[0]) > cal_length(line1[0], line2[1]):
        line2[0], line2[1] = line2[1], line2[0]

    if is_contained_pt(line2, get_foot(line2, line1[0])):
        # 包含
        if is_contained_pt(line2, get_foot(line2, line1[1])):
            overlap = cal_length(line1[0], line1[1])
        elif is_contained_pt(line1, get_foot(line1, line2[1])):
            overlap = cal_length(get_foot(line2, line1[0]), line2[1])
        elif is_contained_pt(line1, get_foot(line1, line2[0])):
            overlap = cal_length(get_foot(line2, line1[0]), line2[0])
 
    elif is_contained_pt(line1, get_foot(line1, line2[0])):
        if is_contained_pt(line1, get_foot(line1, line2[1])):
            overlap = cal_length(line2[0], line2[1])        
        elif is_contained_pt(line2, get_foot(line2, line1[1])):
            overlap = cal_length(get_foot(line1, line2[0]), line1[1])
        elif is_contained_pt(line2, get_foot(line2, line1[0])):
            overlap = cal_length(get_foot(line1, line2[0]), line1[0])

    return overlap

def rbox2poly(obboxes):
    """
    Trans rbox format to poly format.
    Args:
        rboxes (array/tensor): (num_gts, [cx cy l s θ]) θ∈[-pi/2, pi/2)

    Returns:
        polys (array/tensor): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4]) 
    """
    if isinstance(obboxes, torch.Tensor):
        center, w, h, theta = obboxes[:, :2], obboxes[:, 2:3], obboxes[:, 3:4], obboxes[:, 4:5]
        Cos, Sin = torch.cos(theta), torch.sin(theta)

        vector1 = torch.cat(
            (w/2 * Cos, -w/2 * Sin), dim=-1)
        vector2 = torch.cat(
            (-h/2 * Sin, -h/2 * Cos), dim=-1)
        point1 = center + vector1 + vector2
        point2 = center + vector1 - vector2
        point3 = center - vector1 - vector2
        point4 = center - vector1 + vector2
        order = obboxes.shape[:-1]
        return torch.cat(
            (point1, point2, point3, point4), dim=-1).reshape(*order, 8)
    else:
        center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
        Cos, Sin = np.cos(theta), np.sin(theta)

        vector1 = np.concatenate(
            [w/2 * Cos, -w/2 * Sin], axis=-1)
        vector2 = np.concatenate(
            [-h/2 * Sin, -h/2 * Cos], axis=-1)

        point1 = center + vector1 + vector2
        point2 = center + vector1 - vector2
        point3 = center - vector1 - vector2
        point4 = center - vector1 + vector2
        order = obboxes.shape[:-1]
        return np.concatenate(
            [point1, point2, point3, point4], axis=-1).reshape(*order, 8)

def get_O_R(p1, p2, p3):
    '''三点求圆，返回圆心和半径'''

    x, y, z = p1[0]+p1[1]*1j, p2[0]+p2[1]*1j, p3[0]+p3[1]*1j
    w = z-x
    w /= y-x
    c = (x-y)*(w-abs(w)**2)/2j/w.imag-x

    return (-c.real,-c.imag), abs(c+x)

def get_arc_points(center_point, radius, start_angle, end_angle, num_points):
    """
    获取弧形上的点坐标列表
    :param radius: 弧形的半径
    :param start_angle: 起始角度，单位为度
    :param end_angle: 结束角度，单位为度
    :param num_points: 总点数
    :return: 点坐标列表 [(x1, y1), (x2, y2), ...]
    """
    # 计算弧形的圆心坐标
    center_x = center_point[0]
    center_y = center_point[1]

    # 将角度转换为弧度
    start_angle_rad = math.radians(start_angle)
    end_angle_rad = math.radians(end_angle)

    # 计算角度增量
    if abs(end_angle_rad - start_angle_rad) > 6.28:
        angle_increment = abs((abs(end_angle_rad) - abs(start_angle_rad))) / (num_points - 1)
    else:
        angle_increment = (end_angle_rad - start_angle_rad) / (num_points - 1)

    # 循环生成点坐标
    points = []
    for i in range(num_points):
        angle = start_angle_rad + i * angle_increment
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        points.append([x, y])

    return points

def cal_arc_points(start_point, end_point, middle_point, divided_counts=20):
    center_points, r = get_O_R(start_point, end_point, middle_point)

    start_vector = [start_point[0] - center_points[0], start_point[1] - center_points[1]]
    end_vector = [end_point[0] - center_points[0], end_point[1] - center_points[1]]

    start_angle_rad = math.atan2(start_vector[1], start_vector[0])
    end_angle_rad = math.atan2(end_vector[1], end_vector[0])

    start_angle = math.degrees(start_angle_rad)
    end_angle = math.degrees(end_angle_rad)

    arc_points = get_arc_points(center_points, r, start_angle, end_angle, divided_counts)
    distance = Point(middle_point).distance(LineString(arc_points))
    if distance > 5.0:
        angle = 360 - abs(end_angle - start_angle)
        end_angle = angle + start_angle
        # if end_angle >= 0:
        #     end_angle = end_angle - 360
        #     # end_angle = 360 - end_angle
        # else:
        #     end_angle = end_angle + 360
        arc_points = get_arc_points(center_points, r, start_angle, end_angle, divided_counts)

    return arc_points

def point_on_ray(x1, y1, x2, y2, distance):
    """计算在射线上距离起点distance的点坐标"""
    
    # 计算射线的角度
    angle = math.atan2(y2 - y1, x2 - x1)
    # 计算在射线上距离起点distance的点的坐标
    x = x1 + distance * math.cos(angle)
    y = y1 + distance * math.sin(angle)
    return (x, y)

def cal_interior_angle(pt1, pt2, pt3):
    """计算内角"""
    def get_angle(x1, y1, x2, y2):
        # Use dotproduct to find angle between vectors
        # This always returns an angle between 0, pi
        numer = (x1 * x2 + y1 * y2)
        denom = math.sqrt((math.pow(x1, 2) + math.pow(y1, 2)) * (math.pow(x2, 2) + math.pow(y2, 2)))
        return math.acos(numer / denom) / math.pi * 180

    def cross_sign(x1, y1, x2, y2):
        # True if cross is positive
        # False if negative or zero
        return x1 * y2 > x2 * y1

    x1, y1 = pt1[0] - pt2[0], pt1[1] - pt2[1]
    x2, y2 = pt3[0] - pt2[0], pt3[1] - pt2[1]
    angle = get_angle(x1, y1, x2, y2)
    if cross_sign(x1, y1, x2, y2):
        angle = 360 - angle

    return angle

def perpendicular_line(p1, p2, p, length):

    # 计算线段的向量
    v = np.array(p2) - np.array(p1)

    # 计算线段的长度
    segment_length = np.linalg.norm(v)

    # 计算线段的单位向量
    unit_vector = v / segment_length

    # 计算垂线的单位向量
    perpendicular_vector = np.array([-unit_vector[1], unit_vector[0]])

    # 计算垂线的起点和终点
    start = np.array(p) - length / 2 * perpendicular_vector
    end = np.array(p) + length / 2 * perpendicular_vector

    return start, end

if __name__ == "__main__":


    a = [[499, 373], [495, 581]] 
    b = [[495.0, 298.15280775192394], [495.0, 221.88265476339936]]

    # [array([499, 373], dtype=int32), array([495, 581], dtype=int32)] [[499.0, 374.3728649400279], [499.0, 439.3762661717102]]


    print(projection_len(a, b))
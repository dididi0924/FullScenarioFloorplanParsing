import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from postprocess.math_utils import *

from config import OUTER_WALL_WIDTH
from collections import defaultdict

def revise_short_wall(contours):
    """矫正短墙"""

    for i in range(len(contours[:])):
        # 遍历当前轮廓的墙线
        if len(contours[i]) < 8:
            continue
        for j in range(len(contours[i])):
            pt1_idx = j
            pt2_idx = (j + 1) % (len(contours[i]))
            pt3_idx = (j + 2) % (len(contours[i]))
            pt4_idx = (j + 3) % (len(contours[i]))
            pt5_idx = (j + 4) % (len(contours[i]))
            pt6_idx = (j + 5) % (len(contours[i]))

            line1 = (contours[i][pt1_idx][0], contours[i][pt2_idx][0])
            line2 = (contours[i][pt2_idx][0], contours[i][pt3_idx][0])
            line3 = (contours[i][pt3_idx][0], contours[i][pt4_idx][0])
            line4 = (contours[i][pt4_idx][0], contours[i][pt5_idx][0])
            line5 = (contours[i][pt5_idx][0], contours[i][pt6_idx][0])

            if cal_angle(line1, line5) == 0 or cal_angle(line1, line5) == 180:
                if cal_length(line3[0], line3[1]) < 15:
                    # print(line2, line4)
                    # print("angle", cal_angle(line2, line4))
                    if cal_angle(line2, line4) < 32:
                        if abs(line2[0][0] - line2[1][0]) >= abs(line2[0][1] - line2[1][1]):
                            contours[i][pt3_idx][0][1] = line2[0][1]
                            contours[i][pt4_idx][0][0] = line2[1][0]
                            contours[i][pt4_idx][0][1] = line5[0][1]
                        else:
                            contours[i][pt3_idx][0][0] = line2[0][0]
                            contours[i][pt4_idx][0][1] = line2[1][1]
                            contours[i][pt4_idx][0][0] = line5[0][0]

    return contours

def remove_short_walls(contours):
    """删除短墙"""

    for i in range(len(contours[:])):
        # 遍历当前轮廓的墙线
        for j in range(len(contours[i]) - 1, -1, -1):
            pt1_idx = j
            pt2_idx = (j + 1) % (len(contours[i]))
            pt3_idx = (j + 2) % (len(contours[i]))
            pt4_idx = (j + 3) % (len(contours[i]))

            line1 = (contours[i][pt1_idx][0], contours[i][pt2_idx][0])
            line2 = (contours[i][pt2_idx][0], contours[i][pt3_idx][0])
            line3 = (contours[i][pt3_idx][0], contours[i][pt4_idx][0])
            # print(cal_angle(line1, line3), cal_length(line2[0], line2[1]))

            if cal_length(line2[0], line2[1]) < 3:
                if (line1[0][0] == line1[1][0] and line3[0][0] == line3[1][0]) or (line1[0][1] == line1[1][1] and line3[0][1] == line3[1][1]):
                    if line1[0][0] == line1[1][0] and line3[0][0] == line3[1][0]:
                        new_x = round((line1[0][0] + line3[0][0]) / 2)
                        contours[i][pt1_idx][0][0] = new_x
                        contours[i][pt2_idx][0][0] = new_x
                        contours[i][pt3_idx][0][0] = new_x
                        contours[i][pt4_idx][0][0] = new_x
                    else:
                        new_y = round((line1[0][1] + line3[0][1]) / 2)
                        contours[i][pt1_idx][0][1] = new_y
                        contours[i][pt2_idx][0][1] = new_y
                        contours[i][pt3_idx][0][1] = new_y
                        contours[i][pt4_idx][0][1] = new_y

    return contours


def strip_pointed(contours):
    """删除尖角"""

    for i in range(len(contours[:])):
        # 遍历当前轮廓的墙线
        for j in range(len(contours[i]) - 1, -1, -1):
            pt1_idx = (j + 2) % (len(contours[i]))
            pt2_idx = (j + 1) % (len(contours[i]))
            pt3_idx = j
            pt4_idx = (j - 1) % (len(contours[i]))
            pt5_idx = (j - 2) % (len(contours[i]))

            line1 = (contours[i][pt1_idx][0], contours[i][pt2_idx][0])
            line2 = (contours[i][pt2_idx][0], contours[i][pt3_idx][0])
            line3 = (contours[i][pt3_idx][0], contours[i][pt4_idx][0])
            line4 = (contours[i][pt4_idx][0], contours[i][pt5_idx][0])
            strip_flag = False
            if (line1[0][0] == line1[1][0] and line4[0][0] == line4[1][0]) or (line1[0][1] == line1[1][1] and line4[0][1] == line4[1][1]):
                if abs(contours[i][pt2_idx][0][1] - contours[i][pt4_idx][0][1]) < 10:
                    if cal_length(line2[0], line2[1]) < 40 and cal_length(line3[0], line3[1]) < 40:
                        strip_flag = True
                    elif cal_angle(line2, line3) < 20:
                        strip_flag = True
                elif abs(contours[i][pt2_idx][0][1] - contours[i][pt4_idx][0][1]) < 15:
                    if cal_length(line2[0], line2[1]) < 15 and cal_length(line3[0], line3[1]) < 15:
                        strip_flag = True

            if strip_flag:
                if line1[0][0] == line1[1][0] and line4[0][0] == line4[1][0]:
                    new_y = int(line1[1][1] + line4[0][1]) / 2
                    contours[i][pt2_idx][0][1] = new_y
                    contours[i][pt4_idx][0][1] = new_y
                else:
                    new_x = int(line1[1][0] + line4[0][0]) / 2
                    contours[i][pt2_idx][0][0] = new_x
                    contours[i][pt4_idx][0][0] = new_x
                contours[i] = np.delete(contours[i], j, axis=0)

    return contours

def revise_pointed(contours):
    """矫正尖角"""

    for i in range(len(contours)):
        # 遍历当前轮廓的墙线
        for j in range(len(contours[i])):
            if len(contours[i]) < 6:
                continue
            cur_idx = j
            next1_idx = (j + 1) % (len(contours[i]))
            next2_idx = (j + 2) % (len(contours[i]))
            angle = cal_interior_angle(contours[i][cur_idx][0], contours[i][next1_idx][0], contours[i][next2_idx][0])
            len1 = cal_length(contours[i][cur_idx][0], contours[i][next1_idx][0])
            len2 = cal_length(contours[i][next1_idx][0], contours[i][next2_idx][0])
            line1 = [contours[i][cur_idx][0], contours[i][next1_idx][0]]
            if angle < 20 or (angle < 30 and len1 < 40 and len2 < 40) or (angle < 45 and len1 < 20 and len2 < 20) or (angle < 60 and len1 < 15 and len2 < 15):
                if contours[i][cur_idx][0][0] - contours[i][next1_idx][0][0] == 0 or contours[i][cur_idx][0][1] - contours[i][next1_idx][0][1] == 0 or \
                    contours[i][next1_idx][0][0] - contours[i][next2_idx][0][0] == 0 or contours[i][next1_idx][0][1] - contours[i][next2_idx][0][1] == 0: 
                    if abs(contours[i][cur_idx][0][0] - contours[i][next1_idx][0][0]) > abs(contours[i][cur_idx][0][1] - contours[i][next1_idx][0][1]):
                        contours[i][next1_idx][0][1] = contours[i][cur_idx][0][1]
                        contours[i] = np.insert(contours[i], next2_idx, [contours[i][next1_idx][0][0], contours[i][next2_idx][0][1]], axis=0)
                    else:
                        contours[i][next1_idx][0][0] = contours[i][cur_idx][0][0]
                        contours[i] = np.insert(contours[i], next2_idx, [contours[i][next2_idx][0][0], contours[i][next1_idx][0][1]], axis=0)
                else:
                    mid_x = (contours[i][cur_idx][0][0] + contours[i][next2_idx][0][0]) / 2
                    mid_y = (contours[i][cur_idx][0][1] + contours[i][next2_idx][0][1]) / 2
                    mid_line = [[mid_x, mid_y], line1[1]]
                    foot = get_foot(mid_line, line1[0])
                    new_pt_x = contours[i][next1_idx][0][0] + contours[i][cur_idx][0][0] - foot[0]
                    new_pt_y = contours[i][next1_idx][0][1] + contours[i][cur_idx][0][1] - foot[1]
                    contours[i] = np.insert(contours[i], next2_idx, [contours[i][next1_idx][0][0] - contours[i][cur_idx][0][0] + foot[0], contours[i][next1_idx][0][1] - contours[i][cur_idx][0][1] +  foot[1]], axis=0)
                    contours[i][next1_idx][0][0] = new_pt_x
                    contours[i][next1_idx][0][1] = new_pt_y

    return contours


def generate_outer_wall(contours, scale):
    """生成外墙"""

    expand = OUTER_WALL_WIDTH / 10 / scale
    polygon_list = []
    buffer_polygon_list = []
    for contour in contours:
        poly = Polygon(np.squeeze(contour, axis=1))
        polygon_list.append(poly)
        buffer_polygon_list.append(poly.buffer(expand, join_style=2))

    sector = unary_union(buffer_polygon_list)
    outer_contour = []
    try:
        for x, y in zip(sector.exterior.xy[0], sector.exterior.xy[1]):
            outer_contour.append([[x, y]])
    except:
        pass
    
    return np.array(outer_contour, dtype=np.int32)



def revise_by_real_corners(contours, real_corners, dis_thresh=5):
    """矫正角点"""

    for corner in real_corners:
        min_dis = 99
        min_corner = []
        idx_i, idx_j = 0, 0
        for i, contour in enumerate(contours):
            for j, pt in enumerate(contour):
                dis = cal_length(pt[0], corner)
                if dis < dis_thresh and dis < min_dis:
                    min_dis = dis
                    min_corner = corner
                    idx_i = i
                    idx_j = j
        if min_corner != []:
            # print(contours[i][j][0], min_corner)
            contours[idx_i][idx_j][0][0] = min_corner[0]
            contours[idx_i][idx_j][0][1] = min_corner[1]
    
    return contours


def revise_by_wall_line_segments(contours, base_lines):
    """根据直线检测结果矫正墙线"""

    new_contours = []
    all_walls = []
    for i, contour in enumerate(contours[:]):
        cur_walls = []
        for j, pt in enumerate(contour[:]):
            cur_idx = j
            next_idx = (j + 1) % (len(contour))
            cur_walls.append([contour[cur_idx], contour[next_idx]])
        all_walls.append(cur_walls)
    
    for i, contour in enumerate(all_walls):
        cur_contour = []
        for j, wall in enumerate(contour):
            wall_line = [wall[0][0], wall[1][0]]
            wall_len = cal_length(wall_line[0], wall_line[1])
            if wall_len < 10:
                cur_contour.append(wall[0])
                cur_contour.append(wall[1])
                continue
            mid_wall_pt = [(wall_line[0][0] + wall_line[1][0]) / 2, (wall_line[0][1] + wall_line[1][1]) / 2]
            # 遍历基准线
            max_cnt = -1
            min_dis = 99
            min_base_line = []
            for line in base_lines[:]:
                base_line = [[line[0], line[1]], [line[2], line[3]]]
                base_len = cal_length(base_line[0], base_line[1])
                if base_len < 10:
                    continue
                angle = cal_angle(wall_line, base_line)
                if 45 < angle < 135:
                    continue
                
                if cal_pt_line_dis(mid_wall_pt, base_line) > 5:
                    continue

                step = 10
                dis = 99
                for k in range(step):
                    x = wall_line[0][0] + (wall_line[1][0] - wall_line[0][0]) / (step - 1) * k
                    y = wall_line[0][1] + (wall_line[1][1] - wall_line[0][1]) / (step - 1) * k
                    cur_dis = cal_pt_line_dis_2([x, y], base_line)
                    # print([x, y], cur_dis)
                    if cur_dis < 5 and cur_dis < dis:
                        dis = cur_dis
                    # dis += cal_pt_line_dis([x, y], base_line)
                if dis < min_dis:
                    min_dis = dis
                    min_base_line = base_line

            if min_base_line != []:
                foot1 = get_foot(min_base_line, wall_line[0])
                foot2 = get_foot(min_base_line, wall_line[1])

                cur_contour.append([foot1])
                cur_contour.append([foot2])
            else:
                cur_contour.append(wall[0])
                cur_contour.append(wall[1])

        new_contours.append(np.array(cur_contour, dtype=int))


    for i, contour in enumerate(new_contours):
        for j in range(len(contours[i]) - 1, -1, -1):
            pt1_idx = j
            pt2_idx = (j - 1) % (len(contours[i]))
            # print(contours[i][pt1_idx][0], contours[i][pt2_idx][0])
            if (contours[i][pt1_idx][0] == contours[i][pt2_idx][0]).all():
                contours[i] = np.delete(contours[i], pt1_idx, axis=0)
    # for contour in all_walls:
        
    #     for wall in contour:
    #         for pt in wall:
    #             print(pt)
    #             cur_contour.append(pt)
        
    #     new_contours.append(np.array(cur_contour))

    return new_contours



def split_contours(contours, hierarchy):
    """将轮廓分开"""

    contour_ls = []

    contours_dict = defaultdict(list)
    for idx, (contour, h), in enumerate(zip(contours, hierarchy[0])):
        parent = h[3]
        if parent == -1:
            contours_dict[idx] = [idx]
        else:
            for contours_key in contours_dict.keys():
                if parent in contours_dict[contours_key]:
                    contours_dict[contours_key].append(idx)

    for idx_ls in contours_dict.values():
        cur_contours = []
        for idx in idx_ls:
            cur_contours.append(contours[idx])
        contour_ls.append(cur_contours)

    max_contour_area = 0
    max_contour_idx = 0
    for idx, splited_contours in enumerate(contour_ls):
        area = cv2.contourArea(splited_contours[0])
        if area > max_contour_area:
            max_contour_idx = idx
            max_contour_area = area

    return contour_ls[max_contour_idx]

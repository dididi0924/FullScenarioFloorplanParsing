



# @timer
def revise_lsd_line_segments(lines, wall_mask, module_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    mask = cv2.dilate(wall_mask, kernel, iterations=2)
    # if type(lines) == list:
    #     lines = np.array(lines)

    # lines = lines[:, 0:4]

    lines = revise_lines(lines)
    lines = filter_lines(lines, mask)
    # pre_lines = deepcopy(lines)
    # flag = True
    # while flag:
    #     lines, flag = merge_lines(lines)
    #     print(len(lines))
    # lines, flag = merge_lines(lines)
    wall_lines, module_lines = split_lines(lines, module_mask)

    return wall_lines, module_lines

def revise_lines(lines, thresh=2):
    """矫正轻微斜线"""
    
    for i, line in enumerate(lines):
        pt1_x = line[0]
        pt1_y = line[1]
        pt2_x = line[2]
        pt2_y = line[3]

        if 0 < abs(pt1_x - pt2_x) <= thresh:
            mid_x = int(round((pt1_x + pt2_x) / 2))
            lines[i][0] = lines[i][2] = mid_x

        elif 0 < abs(pt1_y - pt2_y) <= thresh:
            mid_y = int(round((pt1_y + pt2_y) / 2))
            lines[i][1] = lines[i][3] = mid_y

    return lines

def filter_lines(lines, wall_mask, step=5):
    """过滤直线"""
    new_lines = []
    for line in lines:
        pt1_x = int(round(line[0]))
        pt1_y = int(round(line[1]))
        pt2_x = int(round(line[2]))
        pt2_y = int(round(line[3]))
        cnt = 0
        for i in range (step):
            x = min(int(pt1_x + (pt2_x - pt1_x) / (step - 1) * i), wall_mask.shape[1] - 1)
            y = min(int(pt1_y + (pt2_y - pt1_y) / (step - 1) * i), wall_mask.shape[0] - 1)
            if wall_mask[y, x].all() == 0:
                cnt += 1
        
        if cnt > step / 2:
            continue
        else:
            new_lines.append(line)

    return new_lines

@timer
def get_corners(lsd_line_segments, dis_thresh=4.5, angle_thresh=15):
    """获取角点"""

    corners = []
    for i, ls1 in enumerate(lsd_line_segments):
        line1 = [[ls1[0], ls1[1]], [ls1[2], ls1[3]]]
        min_dis_1 = 99
        min_line_1 = []
        min_dis_2 = 99
        min_line_2 = []
        for ls2 in lsd_line_segments[i+1:]:
            line2 = [[ls2[0], ls2[1]], [ls2[2], ls2[3]]]
            if angle_thresh <= cal_angle(line1, line2) <= 180 - angle_thresh:
                dis_1 = min(cal_length(line1[0], line2[0]), cal_length(line1[0], line2[1]))
                dis_2 = min(cal_length(line1[1], line2[0]), cal_length(line1[1], line2[1]))
                if dis_1 < dis_thresh and dis_1 < min_dis_1:
                    min_dis_1 = dis_1
                    min_line_1 = line2
                elif dis_2 < dis_thresh and dis_2 < min_dis_2:
                    min_dis_2 = dis_2
                    min_line_2 = line2

        if min_line_1 != []:
            corner = line_intersection(line1, min_line_1)
            corners.append(corner)
        if min_line_2 != []:
            corner = line_intersection(line1, min_line_2)
            corners.append(corner)

    return corners
import cv2
import numpy as np

from shapely.geometry import Polygon, LineString, MultiLineString
from shapely.ops import unary_union

import math
from typing import List, Tuple

import numpy as np
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.ops import unary_union


from shapely.strtree import STRtree  # 引入空间索引

def angle_between_vectors(v1, v2):
    """计算两向量的夹角（弧度），范围 [0, pi]"""
    v1 = v1 / (np.linalg.norm(v1) + 1e-6)
    v2 = v2 / (np.linalg.norm(v2) + 1e-6)
    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(cos_theta)

def point_to_segment_distance(p, a, b):
    """计算点 p 到线段 ab 的最近点和距离"""
    ap = p - a
    ab = b - a
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-6)
    t = np.clip(t, 0, 1)
    proj = a + t * ab
    dist = np.linalg.norm(p - proj)
    return proj, dist

def contours_to_polygons(contours):
    """将OpenCV contours转换为Shapely多边形"""
    polys = []
    for cnt in contours:
        if len(cnt) >= 3:
            pts = cnt[:,0,:]  # (N,2)
            poly = Polygon(pts)
            if poly.is_valid and poly.area > 1:
                polys.append(poly)
    return polys

def polygons_to_contours(polys):
    """将Shapely多边形转换为OpenCV contours"""
    contours = []
    for poly in polys:
        if poly.is_empty:
            continue
        if poly.geom_type == 'Polygon':
            ext = np.array(poly.exterior.coords, dtype=np.int32).reshape(-1,1,2)
            contours.append(ext)
        elif poly.geom_type == 'MultiPolygon':
            for p in poly:
                ext = np.array(p.exterior.coords, dtype=np.int32).reshape(-1,1,2)
                contours.append(ext)
    return contours

def snap_postprocess(contours, lsd_lines, max_dist=5, max_angle=np.pi/6):
    """
    使用 shapely.buffer 收缩轮廓并贴合到 LSD 直线
    参数:
        contours: cv2.findContours得到的轮廓
        lsd_lines: [[x1,y1,x2,y2], ...] 直线检测结果
        max_dist: 最大允许距离（像素）
        max_angle: 最大允许夹角（弧度）
    返回:
        new_contours: 优化后的轮廓
    """
    # --- 1. 轮廓整体收缩 2px ---
    polys = contours_to_polygons(contours)
    if not polys:
        return []

    eroded = [p.buffer(-2) for p in polys]  # 收缩
    eroded = [p for p in eroded if not p.is_empty]
    eroded_contours = polygons_to_contours(eroded)

    new_contours = []

    # --- 2. 遍历每个轮廓点 ---
    for cnt in eroded_contours:
        new_cnt = []
        for i in range(len(cnt)):
            p = cnt[i,0].astype(np.float32)

            # 计算轮廓局部方向
            prev_p = cnt[(i-1)%len(cnt),0].astype(np.float32)
            next_p = cnt[(i+1)%len(cnt),0].astype(np.float32)
            local_dir = next_p - prev_p

            best_proj = p
            best_score = 1e9

            # 遍历所有 LSD 线段
            for (x1,y1,x2,y2) in lsd_lines:
                a = np.array([x1,y1], np.float32)
                b = np.array([x2,y2], np.float32)
                line_dir = b - a

                # 角度约束
                ang = angle_between_vectors(local_dir, line_dir)
                if ang > np.pi/2:
                    ang = np.pi - ang

                if ang < max_angle:
                    proj, dist = point_to_segment_distance(p, a, b)
                    if dist < max_dist:
                        score = dist + ang*2
                        if score < best_score:
                            best_score = score
                            best_proj = proj

            new_cnt.append(best_proj)

        new_cnt = np.array(new_cnt, dtype=np.int32).reshape(-1,1,2)
        new_contours.append(new_cnt)

    return new_contours



def find_max_rectangle(segment1, segment2):
    """
    Find the largest rectangle formed by the normal direction of two parallel line segments.
    segment1 and segment2 are lists of two points each: [(x1, y1), (x2, y2)].
    Returns the rectangle as four vertices [p1, p2, p3, p4] in order, or None if no rectangle exists.
    """
    # Extract points
    A = np.array(segment1[0])
    B = np.array(segment1[1])
    C = np.array(segment2[0])
    D = np.array(segment2[1])
    
    # Check if segments are parallel (vectors AB and CD must be parallel)
    AB = B - A
    CD = D - C
    if np.cross(AB, CD) != 0:
        return None  # Segments are not parallel
    
    # Normalize direction vector
    direction = AB / np.linalg.norm(AB)
    normal = np.array([-direction[1], direction[0]])  # Perpendicular unit vector
    
    # Project all points onto the direction and normal axes
    # Parallel projections (scalar values along direction)
    A_para = np.dot(A, direction)
    B_para = np.dot(B, direction)
    C_para = np.dot(C, direction)
    D_para = np.dot(D, direction)
    
    # Normal projections (scalar values along normal)
    A_norm = np.dot(A, normal)
    B_norm = np.dot(B, normal)
    C_norm = np.dot(C, normal)
    D_norm = np.dot(D, normal)
    
    # Check overlap in parallel direction
    min_para = max(min(A_para, B_para), min(C_para, D_para))
    max_para = min(max(A_para, B_para), max(C_para, D_para))
    if min_para >= max_para:
        return None  # No overlap in parallel direction
    
    # Normal range (min and max of all normal projections)
    min_norm = min(A_norm, B_norm, C_norm, D_norm)
    max_norm = max(A_norm, B_norm, C_norm, D_norm)
    
    # Calculate rectangle vertices
    # Start with the two points along the direction axis at min_para and max_para
    p1 = min_para * direction + min_norm * normal
    p2 = max_para * direction + min_norm * normal
    p3 = max_para * direction + max_norm * normal
    p4 = min_para * direction + max_norm * normal
    
    # Convert to list of tuples
    rectangle = [p1.tolist(), p2.tolist(), p3.tolist(), p4.tolist()]
    
    return rectangle


def point_dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def vec(a, b):
    return np.array([b[0] - a[0], b[1] - a[1]], dtype=float)


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """返回两向量夹角（弧度，0..pi）。若任一向量长度接近0，返回 pi/2。"""
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6:
        return math.pi / 2
    cos = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
    return math.acos(cos)


def project_point_to_segment(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]):
    """将点 p 投影到线段 ab 上，返回 (proj_point, t, dist)，其中 t 为投影在 ab 上的参数（0..1）。"""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    p = np.array(p, dtype=float)
    ab = b - a
    ab2 = np.dot(ab, ab)
    if ab2 == 0:
        return tuple(a), 0.0, np.linalg.norm(p - a)
    t = np.dot(p - a, ab) / ab2
    t_clamped = max(0.0, min(1.0, t))
    proj = a + t_clamped * ab
    dist = np.linalg.norm(p - proj)
    return tuple(proj), float(t_clamped), float(dist)



def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6:
        return math.pi / 2
    cos = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
    return math.acos(cos)


# 把contours和未成功贴附线的点拎出来，绘制到过程图上
def find_bad_contour_points(contours, wall_lines, det_boxes=[], max_dis=20):
    bad_pts = []
    cut_points_number = 100

    # line_objs = MultiLineString([LineString([(x1, y1), (x2, y2)]) for (x1, y1, x2, y2, color) in wall_lines])

    det_polygons = [Polygon(db[11]) for db in det_boxes]
    line_objs = [LineString([(x1, y1), (x2, y2)]) for (x1, y1, x2, y2, _) in wall_lines] + det_polygons

    line_tree = STRtree(line_objs)

    for contour in contours:
        contour_points = contour.reshape(-1, 2)

        pts = [Polygon(contour_points).exterior.interpolate(t, normalized=True) 
                           for t in np.linspace(0, 1, cut_points_number * 2 if Polygon(contour_points).length > 1000 else cut_points_number, endpoint=True)]
        
        # 判断pts是否在wall_lines附近
        for pt in pts:
            # 使用空间索引加速查询
            nearest_line_idx = line_tree.nearest(pt)
            if hasattr(nearest_line_idx, 'item'):  # 如果是 numpy.int64
                nearest_line_idx = nearest_line_idx.item()

            nearest_line = line_objs[nearest_line_idx]  # 获取实际的 LineString
            
            # 计算点到最近墙线的距离
            min_dist = pt.distance(nearest_line)
            if min_dist > max_dis:
                bad_pts.append((int(pt.x), int(pt.y)))

            # nearest_line = line_objs.interpolate(line_objs.project(pt))
            # if pt.distance(nearest_line) > max_dis:
            #     bad_pts.append((int(pt.x), int(pt.y)))

    return bad_pts

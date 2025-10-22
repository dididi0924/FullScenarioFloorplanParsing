import cv2
import random
import numpy as np


from shapely.geometry import GeometryCollection
# from shapely.geometry import Point, Polygon, LineString

def random_color():
    levels = range(32, 256, 32)
    return tuple(random.choice(levels) for _ in range(3))

def shapely_show(objects, save_path=None, equal=False, numbers=False):
    import matplotlib.pyplot as plt
    def get_random_color():
        """获取一个随机的颜色"""
        col = (np.random.random(), np.random.random(), np.random.random())
        return col
    
    def draw_number(number_text, object):
        # 中心绘制编号
        x_center = (object.bounds[0] + object.bounds[2]) / 2
        y_center = (object.bounds[1] + object.bounds[3]) / 2
        ax.text(x_center, y_center, str(number_text), fontsize=12, color='red', ha='center', va='center')

    # 计数
    if numbers:
        number_text = 0

    plt.clf()
    
    # 设置大尺寸和高DPI
    fig = plt.figure(figsize=(38.4, 21.6), dpi=100)  # 3840×2160 (4K)
    ax = fig.add_subplot(111)
    ax.invert_yaxis()
    
    # 老方法
    # ax = plt.gca()
    # ax.invert_yaxis()

    if isinstance(objects, list) == False and isinstance(objects, GeometryCollection) == False:
        objects = [objects]
    for i_object in objects:
        if i_object.type in ['Polygon']:
            # 获取polygon的中心
            

            if len(i_object.interiors) == 0:
                x, y = i_object.exterior.xy
                ax.plot(x, y, color=get_random_color())
            else:
                x, y = i_object.exterior.xy
                ax.fill(x, y, color=get_random_color())

                for interior in i_object.interiors:
                    x, y = interior.xy
                    ax.fill(x, y, color='w')

            if numbers:
                draw_number(number_text, i_object)
                number_text += 1
            

        elif i_object.type in ['LineString']:
            x, y = i_object.xy
            ax.plot(x, y)
        elif i_object.type in ['LinearRing']:
            x, y = i_object.xy
            ax.plot(x, y)
        # 分开 避免颜色一致
        elif i_object.type in ['Point']:
            i_object = i_object.buffer(1)
            x, y = i_object.exterior.xy
            ax.plot(x, y)
        elif i_object.type in ['MultiLineString', ]:
            for line in list(i_object.geoms):
                x, y = line.xy
                ax.plot(x, y)
        elif i_object.type in ['MultiPolygon', ]:
            for polygon in list(i_object.geoms):
                x, y = polygon.exterior.xy
                ax.plot(x, y)

                if numbers:
                    draw_number(number_text, polygon)
                    number_text += 1
        else:
            continue
    if save_path is None:
        plt.axis('equal')
        plt.show()
    else:
        if equal:
            plt.axis('equal')
            
        plt.savefig(save_path)

        # plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.1, quality=100)
        # plt.close()

def draw_contours_random_color(img, contours, line_scale=2):
    for contour in contours:
        cv2.drawContours(img, [contour], -1, random_color(), line_scale)

    return img
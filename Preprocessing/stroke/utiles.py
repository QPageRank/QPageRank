import math

# import dgl
import numpy as np
from math import sin, asin, cos, radians, fabs, sqrt, degrees, atan2

# from py2neo import NodeMatcher, Relationship, Graph
from shapely.geometry import LineString, Polygon

# import torch


def merge_road_points(main_road_p, adjoin_road_p, adjoin_type):
    """
    把两条道路坐标点连接为一条
    :param main_road_p: 主道路坐标点集
    :param adjoin_road_p: 要连接的道路点集
    :param adjoin_type: 邻接类型
    :return: 返回一个新的道路点集
    """
    if adjoin_type == 1:
        del main_road_p[0]
        new_p = adjoin_road_p + main_road_p
    elif adjoin_type == 2:
        del main_road_p[0]
        new_p = list(reversed(adjoin_road_p)) + main_road_p
    elif adjoin_type == 3:
        del main_road_p[-1]
        new_p = main_road_p + adjoin_road_p
    else:
        del main_road_p[-1]
        new_p = main_road_p + list(reversed(adjoin_road_p))
    return new_p


def touchable(line_a, line_b):
    """
    判断俩条道路是否邻接，并判断邻接的类型
    :param line_a: 道路a的几何对象
    :param line_b: 道路b的几何对象
    :return:
    """
    start_point_a = line_a[0]
    end_point_a = line_a[-1]
    start_point_b = line_b[0]
    end_point_b = line_b[-1]

    # a,b首尾相连
    if start_point_a == end_point_b:
        return 1
    # a,b首首相连
    elif start_point_a == start_point_b:
        return 2
    # a,b尾首相连
    elif end_point_a == start_point_b:
        return 3
    # 尾尾相连
    elif end_point_a == end_point_b:
        return 4
    # 不相连
    else:
        return 0


#


def calculate_angele(r_a_p, r_b_p, adjoin_type):
    """
    计算俩条邻接道路夹角，范围在【0，180】度之间，如果夹角为0表示两条道路重合
    如果夹角为180表示两条道路反向平行，应当构建为stroke，如果夹角为90，表示垂直
    :param r_a_p:道路a的坐标点集
    :param r_b_p:道路b的坐标点集
    :param adjoin_type:邻接类型
    :return:返回两条道路的夹角度数
    """
    if adjoin_type == 1:
        node = r_a_p[0]  # 取a的起点或者b的终点作为两条线的连接节点
        point_a = r_a_p[1]  # 取a的正数第二个坐标点
        point_b = r_b_p[-2]  # 取b的倒数第二个点

    elif adjoin_type == 2:
        node = r_a_p[0]  # 取a的起点或者b的终点作为两条线的连接节点
        point_a = r_a_p[1]  # 取a的正数第二个坐标点
        point_b = r_b_p[1]  # 取正数第二个点

    elif adjoin_type == 3:
        node = r_a_p[-1]  # 取a的终点或者b的起点作为两条线的连接节点
        point_a = r_a_p[-2]  # 取a的倒数数第二个坐标点
        point_b = r_b_p[1]  # 取正数第二个点

    elif adjoin_type == 4:
        node = r_a_p[-1]  # 取a的终点或者b的起点作为两条线的连接节点
        point_a = r_a_p[-2]  # 取a的倒数数第二个坐标点
        point_b = r_b_p[-2]  # 取b的倒数第二个点
    else:
        raise ValueError("Invalid adjoin type")
        # 根据道路邻接点和邻接点左右坐标点，构造向量vector_a,vector_b,向量起点为node
    vector_a = np.array(list(map(lambda x: x[1] - x[0], zip(node, point_a))))
    vector_b = np.array(list(map(lambda x: x[1] - x[0], zip(node, point_b))))

    # 计算vector_a,vector_b夹角的余弦值
    ss_a = np.sum(np.square(vector_a))
    ss_b = np.sum(np.square(vector_b))
    cos_theta = np.dot(vector_a, vector_b) / np.sqrt(ss_a * ss_b)
    cos_theta = max(min(cos_theta, 1), -1)
    angle = round(math.degrees(math.acos(cos_theta)))  # round 表示浮点数的四舍五入
    # if -1 <= cos_theta <= 1:
    #     angle = round(math.degrees(math.acos(cos_theta)))#round 表示浮点数的四舍五入
    # else:
    #     angle=120
    return angle

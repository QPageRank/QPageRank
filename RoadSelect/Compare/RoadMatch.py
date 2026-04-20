import os
import sys
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box
import copy

sys.path.append(os.getcwd())
from Tool.JsonTool import read, save

# ========== 参数默认值 ========== #
default_epsg = 4326  # 若无定义时默认 CRS
project_epsg = 3857  # 测量投影 CRS，用于长度/角度计算
BUFFER_DIST = 100.0  # shp2 缓冲区宽度（米）
DIRECTION_TOL = 30  # 方向误差容限（度）


# ========== 工具函数 ========== #
def ensure_crs(gdf):
    if gdf.crs is None:
        gdf.set_crs(epsg=default_epsg, inplace=True)
    if gdf.crs.is_geographic:
        return gdf.to_crs(epsg=project_epsg)
    return gdf


def preprocess_gdf(gdf):
    """保留 orig_id，explode MultiLineString，过滤 LineString"""
    gdf["orig_id"] = gdf.index
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    return gdf[gdf.geometry.type == "LineString"].reset_index(drop=True)


def point_direction(idx, coords):
    if idx < len(coords) - 1:
        p0, p1 = coords[idx], coords[idx + 1]
    else:
        p0, p1 = coords[idx - 1], coords[idx]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    return np.degrees(np.arctan2(dy, dx)) % 180


def tangent_direction(point, line):
    proj = line.project(point)
    d = 1e-3
    p1 = line.interpolate(max(proj - d, 0))
    p2 = line.interpolate(min(proj + d, line.length))
    dx, dy = p2.x - p1.x, p2.y - p1.y
    return np.degrees(np.arctan2(dy, dx)) % 180


def match_roads(gdf1, gdf2, buffer_dist, dir_tol):
    """
    根据缓冲区+方向匹配规则，返回匹配的 orig_id 集合。
    gdf1, gdf2 均应为 LineString 数据。
    """
    # 构建 gdf2 缓冲区和空间索引
    gdf2_buf = gdf2.copy()
    gdf2_buf["geometry"] = gdf2_buf.geometry.buffer(buffer_dist)
    sindex2 = gdf2_buf.sindex

    matched_ids = set()
    # 遍历 gdf1 的每条道路
    for _, row in gdf1.iterrows():
        coords = list(row.geometry.coords)
        for vidx, coord in enumerate(coords):
            pt = Point(coord)
            # 候选缓冲区
            cand = list(sindex2.intersection(pt.bounds))
            for j in cand:
                if gdf2_buf.loc[j].geometry.contains(pt):
                    dir1 = point_direction(vidx, coords)
                    dir2 = tangent_direction(pt, gdf2.loc[j].geometry)
                    diff = abs(dir1 - dir2) % 180
                    if diff > 90:
                        diff = 180 - diff
                    if diff <= dir_tol:
                        matched_ids.add(row["orig_id"])
                        break
            else:
                continue
            break
    return matched_ids


def compute_stats(gdf1, matched_ids):
    total_len = gdf1.geometry.length.sum()
    matched_len = gdf1[gdf1["orig_id"].isin(matched_ids)].geometry.length.sum()
    percent = matched_len / total_len * 100 if total_len > 0 else 0
    return total_len, matched_len, percent


# ========== 主函数 ========== #


def roadMatch(shp1_path, gdf2):
    # 读取文件
    gdf1 = gpd.read_file(shp1_path)
    # gdf2 = gpd.read_file(shp2_path)
    # CRS 处理
    gdf1 = ensure_crs(gdf1)
    # gdf2 = ensure_crs(gdf2)

    # 预处理：explode + 保留 LineString
    gdf1 = preprocess_gdf(gdf1)
    gdf2 = preprocess_gdf(gdf2)

    # 计算 gdf1 的 bounds 并按 BUFFER_DIST 拓展
    minx, miny, maxx, maxy = gdf1.total_bounds
    # 注意：这里用同样的投影（project_epsg）下的米单位
    bbox = box(
        minx - BUFFER_DIST, miny - BUFFER_DIST, maxx + BUFFER_DIST, maxy + BUFFER_DIST
    )
    # 只保留 gdf2 中和这个 bbox 相交的要素
    gdf2 = gdf2[gdf2.intersects(bbox)].copy()
    gdf2 = gdf2.reset_index(drop=True)

    # 计算匹配
    matched_ids = match_roads(gdf1, gdf2, BUFFER_DIST, DIRECTION_TOL)
    print(f"[INFO] 匹配 shp1 道路数: {len(matched_ids)} 条")

    # 统计结果
    total_len, matched_len, percent = compute_stats(gdf1, matched_ids)
    print(f"[RESULT] shp1 总长度: {total_len:.2f} 米")
    print(f"[RESULT] 匹配道路总长度: {matched_len:.2f} 米, 占比: {percent:.2f}%")
    print(f"[RESULT] 匹配道路 orig_id 列表: {sorted(matched_ids)}")

    return matched_ids, total_len, matched_len, percent


def run_RoadMatch(index):
    configpath = os.getcwd() + "/config.json"
    config = read(configpath)

    roadpath = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/SelectResult"
    )
    comparepath = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/compare/compareroad.shp"
    )
    cover_path = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/Result/cover.json"
    )
    key_list = ["quantum", "classic", "GNN"]
    ratio_list = ["0.4"]
    alpha_list = ["0.85"]
    noise_list = ["0"]

    key_map = {}
    for keyname in key_list:
        ratio_map = {}
        for ratio_value in ratio_list:
            alpha_map = {}
            for alpha_value in alpha_list:
                noise_map = {}
                for noise_value in noise_list:
                    road_shp_path = (
                        roadpath
                        + "/"
                        + keyname
                        + "_"
                        + str(ratio_value)
                        + "_"
                        + str(alpha_value)
                        + "_"
                        + str(noise_value)
                        + ".shp"
                    )
                    gdf2 = gpd.read_file(comparepath)
                    gdf2 = ensure_crs(gdf2)
                    _, _, _, percent = roadMatch(road_shp_path, gdf2)
                    noise_map[noise_value] = percent
                    alpha_map[alpha_value] = noise_map
                    ratio_map[ratio_value] = alpha_map
                    key_map[keyname] = ratio_map
    save(cover_path, key_map)


if __name__ == "__main__":
    index = 110
    run_RoadMatch(index)

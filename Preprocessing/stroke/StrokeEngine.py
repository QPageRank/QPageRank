import os
import sys
import shapefile
from Preprocessing.stroke.utiles import (
    touchable,
    calculate_angele,
    merge_road_points,
    LineString,
)
import geopandas as gpd
from itertools import combinations
import chardet

sys.path.append(os.getcwd())
from Tool.JsonTool import read


def detect_encoding(file_path):
    with open(file_path, "rb") as file:
        return chardet.detect(file.read())["encoding"]


def stroke(file_path, outpath, angle_threshold=60):
    # encoding = detect_encoding(file_path.split(".")[0] + ".cpg")
    encoding = "utf-8"
    path = shapefile.Reader(file_path, encoding=encoding)
    # 使用 geopandas 读取原始文件以获取 CRS（如果有）
    original_gdf = gpd.read_file(file_path)
    crs = original_gdf.crs  # 可能是 None 或 CRS 对象
    fields = path.fields
    fields = fields[1:]  # 读取所有字段
    fields_name = [field[0] for field in fields]  # 读取标题字段
    merge_fields_name = fields_name
    merge_fields_number = [
        fields_name.index(field) for field in merge_fields_name
    ]  # 计算了下标，具体？
    roads_points = []  # 坐标集
    roads_records = []  # 字段属性集

    for road in path.shapeRecords():
        roads_points.append(road.shape.points)
        record = []
        for i in merge_fields_number:
            # if isinstance(road.record[i], bytes):
            record.append(str(road.record[i]).strip())  # strip除去字符串两边的空格
            # record.append((road.record[i]))  #这里做修改方便判断类型
        roads_records.append(record)
    index = list(range(len(roads_points)))  # 将坐标集数字化

    cols = merge_fields_name.copy()
    cols.append("geometry")

    # 以下修改是让判断类型不报错
    cols_type = []
    for i in merge_fields_number:
        cols_type.append(type(roads_records[0][i]))
    gdf = gpd.GeoDataFrame(columns=cols).set_geometry("geometry")

    while index:
        len1 = len(index)
        if len1 == 781:
            a = 1
        print(len1)
        main_index = index[0]
        main_points = roads_points[main_index]
        main_record = roads_records[main_index]
        merged = False  # 标记是否成功合并
        pre_join_list = [main_index]
        angle_list = []
        # 获取前向连接的待连接路段
        for join_index in index[1:]:
            join_points = roads_points[join_index]
            adjoin_type = touchable(main_points, join_points)
            if adjoin_type in [1, 2]:
                pre_join_list.append(join_index)
        pre_join_award = list(combinations(pre_join_list, 2))

        # 获取所有前向角度符合的角度与索引
        for i in pre_join_award:
            record1 = roads_records[i[0]]
            record2 = roads_records[i[1]]
            line1 = roads_points[i[0]]
            line2 = roads_points[i[1]]
            if record1 == record2:
                adjoin_type = touchable(line1, line2)
                angle = calculate_angele(line1, line2, adjoin_type)
                angle_list.append([i[0], i[1], angle])
                print()
        #
        if angle_list != []:
            max_angle_list = max(
                angle_list, key=lambda row: row[2] if len(row) > 2 else float("-inf")
            )
            if max_angle_list[2] >= angle_threshold:
                adjoin_type = touchable(
                    roads_points[max_angle_list[0]], roads_points[max_angle_list[1]]
                )
                main_points = merge_road_points(
                    roads_points[max_angle_list[0]],
                    roads_points[max_angle_list[1]],
                    adjoin_type,
                )
                index.remove(max_angle_list[1])
                roads_points[max_angle_list[0]] = main_points
                merged = True
        if merged:
            continue

        pre_join_list = [main_index]
        angle_list = []

        # 获取后向连接的待连接路段
        for join_index in index[1:]:
            join_points = roads_points[join_index]
            adjoin_type = touchable(main_points, join_points)
            if adjoin_type in [3, 4]:
                pre_join_list.append(join_index)
        pre_join_back = list(combinations(pre_join_list, 2))

        # 获取所有后向角度符合的角度与索引
        for i in pre_join_back:
            record1 = roads_records[i[0]]
            record2 = roads_records[i[1]]
            line1 = roads_points[i[0]]
            line2 = roads_points[i[1]]
            if record1 == record2:
                adjoin_type = touchable(line1, line2)
                angle = calculate_angele(line1, line2, adjoin_type)
                angle_list.append([i[0], i[1], angle])
                print()
        #
        if angle_list != []:
            max_angle_list = max(
                angle_list, key=lambda row: row[2] if len(row) > 2 else float("-inf")
            )
            if max_angle_list[2] >= angle_threshold:
                adjoin_type = touchable(
                    roads_points[max_angle_list[0]], roads_points[max_angle_list[1]]
                )
                main_points = merge_road_points(
                    roads_points[max_angle_list[0]],
                    roads_points[max_angle_list[1]],
                    adjoin_type,
                )
                index.remove(max_angle_list[1])
                roads_points[max_angle_list[0]] = main_points
                merged = True

        # 如果未成功合并，强制移除main_index
        if not merged:
            index.remove(index[0])
            gdf.loc[main_index, merge_fields_name] = main_record
            gdf.loc[main_index, "geometry"] = LineString(main_points)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    gdf.crs = crs
    gdf.to_file(
        outpath,
        encoding=encoding,
    )
    print("写入成功")


def run_stroke(index):
    configpath = os.getcwd() + "/config.json"
    config = read(configpath)
    sample_shp = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/sample/"
        + "/road_sample"
        + ".shp"
    )
    out_shp = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/stroke/"
        + "/road_stroke"
        + ".shp"
    )
    stroke(sample_shp, out_shp)


if __name__ == "__main__":
    index = 9
    run_stroke(index)

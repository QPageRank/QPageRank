import os
import sys
import json
import chardet
import geopandas as gpd


sys.path.append(os.getcwd())
from Tool.JsonTool import read, save


def detect_encoding(file_path):
    with open(file_path, "rb") as file:
        return chardet.detect(file.read())["encoding"]


def point_row(file, row):
    file.seek(0)
    for i in range(row):
        next(file)


def split(line):
    line = line.strip()
    row_data = line.split(",")
    return row_data


def get_appointline(file, line_number):
    point_row(file, line_number - 1)
    for current_line_number, nodedata_line in enumerate(file, start=line_number):
        row_nodedata = split(nodedata_line)
        return current_line_number, row_nodedata


def safe_float_conversion(s):
    try:
        return float(s)
    except ValueError:
        return 0.0


def read_nodedata(nodedata_path):
    nodedata_map = {}
    length_sum = 0
    nodedata_encoding = detect_encoding(nodedata_path)
    with open(nodedata_path, "r", encoding=nodedata_encoding) as nodedata_file:
        current_line_number, row_nodedata = get_appointline(nodedata_file, 1)
        index = 0
        for nodedata in row_nodedata:
            if nodedata == "Shape_Length":
                break
            index += 1

        for node_index, node_line in enumerate(nodedata_file, start=2):
            row_nodedata = split(node_line)
            value = safe_float_conversion(row_nodedata[index])
            length_sum += value
            nodedata_map[row_nodedata[0]] = value

    return nodedata_map, length_sum


def select_id(data_map, nodedata_map, length_select):
    result_map = {}
    for key1, value1 in data_map.items():
        child_map = {}
        for key2, value2 in value1.items():
            object_id_list = json.loads(value2["sortindex"])
            length = 0
            selectid_list = []
            for object_id in object_id_list:
                selectid_list.append(object_id)
                length += nodedata_map[str(object_id)]
                if length >= length_select:
                    break
            child_map[key2] = str(selectid_list)
        result_map[key1] = child_map
    return result_map


def select_id_2(data_map, nodedata_map, length_select):
    result_map = {}
    child_map = {}
    for key1, value1 in data_map.items():
        object_id_list = json.loads(value1["sorted"])
        length = 0
        selectid_list = []
        for object_id in object_id_list:
            selectid_list.append(object_id)
            length += nodedata_map[str(object_id)]
            if length >= length_select:
                break
        child_map[key1] = str(selectid_list)
    result_map["0.85"] = child_map
    return result_map


def select_ratio(
    nodedata_path,
    class_path,
    quantum_path,
    gnn_path,
    selectresult_path,
    zoomratio,
    gdb_path,
    feature_layer_name,
    outpath,
):
    result_map = {}
    class_map = read(class_path)
    quantum_map = read(quantum_path)
    gnn_map = read(gnn_path)

    nodedata_map, length_sum = read_nodedata(nodedata_path)

    class_select = {}
    quantum_select = {}
    gnn_select = {}
    for ratio in zoomratio:
        length_select = length_sum * ratio
        class_select[ratio] = select_id(class_map, nodedata_map, length_select)
        quantum_select[ratio] = select_id(quantum_map, nodedata_map, length_select)
        gnn_select[ratio] = select_id_2(gnn_map, nodedata_map, length_select)

    # result_map["classic"] = class_select
    # result_map["quantum"] = quantum_select
    result_map["GNN"] = gnn_select
    outroadshp(gdb_path, feature_layer_name, result_map, outpath)
    # outroadshp_gnn(gdb_path, feature_layer_name, gnn_select, outpath)

    save(selectresult_path, result_map)


def outroadshp(gdb_path, feature_layer_name, result_map, outpath):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    gdf = gpd.read_file(gdb_path, layer=feature_layer_name)
    for class_key, class_value in result_map.items():
        for ratio_key, ratio_value in class_value.items():
            for alpha_key, alpha_value in ratio_value.items():
                for noise_key, noise_value in alpha_value.items():
                    original_list = json.loads(noise_value)
                    id_list = [x - 1 for x in original_list]
                    filtered = gdf[
                        (gdf.geometry.type == "MultiLineString")
                        & (gdf.index.isin(id_list))
                    ]
                    output_shp_path = (
                        outpath
                        + "/"
                        + class_key
                        + "_"
                        + str(ratio_key)
                        + "_"
                        + str(alpha_key)
                        + "_"
                        + str(noise_key)
                        + ".shp"
                    )
                    filtered.to_file(output_shp_path)


def outroadshp_gnn(gdb_path, feature_layer_name, gnn_select, outpath):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    gdf = gpd.read_file(gdb_path, layer=feature_layer_name)
    for ratio_key, ratio_value in gnn_select.items():
        for noise_key, noise_value in ratio_value.items():
            original_list = json.loads(noise_value)
            id_list = [x - 1 for x in original_list]
            filtered = gdf[
                (gdf.geometry.type == "MultiLineString") & (gdf.index.isin(id_list))
            ]
            output_shp_path = outpath + "/GNN_" + str(ratio_key) + "_0.85_0.shp"
            filtered.to_file(output_shp_path)


def run_SelectRate(index):
    configpath = os.getcwd() + "/config.json"
    config = read(configpath)

    class_path = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/Result/PageRankWeightResult.json"
    )
    quantum_path = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/Result/QPageRankWeightResult.json"
    )
    gnn_path = os.getcwd() + "/Data/RoadSelect/" + str(index) + "/Result/GNNResult.json"
    selectresult_path = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/Result/selectresult.json"
    )
    nodedata_path = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/txt/nodedata"
        + str(index)
        + ".txt"
    )
    zoomratio = config["roadselect"]["resultprocess"]["zoomratio"]

    gdb_path = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/Roadgdbdata/"
        + str(index)
        + ".gdb"
    )
    feature_layer_name = "road_" + str(index)
    outpath = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/SelectResult"
    )

    select_ratio(
        nodedata_path,
        class_path,
        quantum_path,
        gnn_path,
        selectresult_path,
        zoomratio,
        gdb_path,
        feature_layer_name,
        outpath,
    )


if __name__ == "__main__":
    index = 116
    run_SelectRate(index)

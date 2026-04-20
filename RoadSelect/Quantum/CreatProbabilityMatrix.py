import numpy as np
import os
import networkx as nx
import chardet
import math


# 代码中概率转移矩阵为列到行，邻接矩阵由状态转移矩阵得到，所以二者逻辑相同
class CreatProbabilityMatrix:
    strok_filename = ""
    nodedata_filename = ""
    ProbabilityMatrix = None
    AdjacencyMatrix = None
    nodedata_map = {}
    stroke_map = {}
    node_att = {}

    def __init__(
        self,
        strok_filename,
        nodedata_filename,
        noise_type="attribute",
        noise=0,
        att_list=["Shape_Length"],
    ):
        self.strok_filename = strok_filename
        self.nodedata_filename = nodedata_filename
        self.att_list = att_list
        self.noise_type = noise_type
        self.noise = noise
        self.transform_to_adjacency()

    # 识别文件编码
    def detect_encoding(self, file_path):
        with open(file_path, "rb") as file:
            return chardet.detect(file.read())["encoding"]

    # 分割每一行元素
    def split(self, line):
        line = line.strip()
        row_data = line.split(",")
        return row_data

    # 跳过行数
    def point_row(self, file, row):
        file.seek(0)
        for i in range(row):
            next(file)

    # 读取指定行数
    def get_appointline(self, file, line_number):
        self.point_row(file, line_number - 1)
        for current_line_number, nodedata_line in enumerate(file, start=line_number):
            row_nodedata = self.split(nodedata_line)
            return current_line_number, row_nodedata

    # 计算出度的权重
    def calculate_out_weight(self, strok_file, nodedata_file, node_index, start_line):
        child_nodes = []
        child_weight = 0
        self.point_row(strok_file, start_line - 1)
        for strok_index, strok_line in enumerate(strok_file, start=start_line):
            row_strokdata = self.split(strok_line)
            if row_strokdata[1] == node_index:
                current_line_number, row_nodedata = self.get_appointline(
                    nodedata_file, int(row_strokdata[2]) + 1
                )
                child_nodes.append((current_line_number - 1, float(row_nodedata[2])))
                child_weight += float(row_nodedata[2])
            else:
                return child_nodes, child_weight, strok_index
        return child_nodes, child_weight, strok_index

    def normalize_min_max(self, data):
        # 计算最小值和最大值
        min_val = min(data)
        max_val = max(data)

        # 避免除以零的情况
        if max_val == min_val:
            return [0.0] * len(data)  # 或者返回 [1.0] * len(data)，取决于需求

        # 应用归一化公式
        return [(x - min_val) / (max_val - min_val) for x in data]

    def normalize_sum(self, data):
        # 将数据转换为 NumPy 数组（如果输入是列表）
        data = np.array(data)

        # 计算数据的总和
        num = np.sum(data)

        # 避免除以零的情况
        if np.isclose(num, 0.0, atol=1e-15):  # 使用 NumPy 的 isclose 方法判断是否接近零
            return np.zeros_like(data)  # 返回全零数组

        # 应用归一化公式
        return data / num

    def safe_float_conversion(self, s):
        try:
            return float(s)
        except ValueError:
            return 0.0

    def calculate_similarity(self, nodedata_map, featuresum_map, objectid):
        similarity_map = {}
        for i in range(len(objectid)):
            sum_node = 0
            if self.att_list == []:
                for feature_name, feature_values in nodedata_map.items():
                    sum_node += (feature_values[i] - featuresum_map[feature_name]) ** 2
            else:
                for feature_name in self.att_list:
                    feature_values = nodedata_map[feature_name]
                    sum_node += (feature_values[i] - featuresum_map[feature_name]) ** 2
            similarity_map[objectid[i]] = math.sqrt(sum_node)
        return self.measure_sim(similarity_map)

    def measure_sim(self, similarity_map):
        if similarity_map:
            D_max = max(similarity_map.values())
        else:
            D_max = 0  # 如果字典为空，可以设置一个默认值

        # 遍历字典并替换值
        for key in similarity_map:
            D_i = similarity_map[key]
            S_i = 1 - (D_i / D_max)
            similarity_map[key] = S_i
        return similarity_map

    # 构建概率转移矩阵
    def creat_probability_matrix(self):
        strok_encoding = self.detect_encoding(self.strok_filename)
        nodedata_encoding = self.detect_encoding(self.nodedata_filename)
        index_node_map = {}
        featuresum_map = {}
        objectid = []
        nodenum = 0

        with open(
            self.nodedata_filename, "r", encoding=nodedata_encoding
        ) as nodedata_file:
            nodenum = sum(1 for line in nodedata_file) - 1
            current_line_number, row_nodedata = self.get_appointline(nodedata_file, 1)
            index = 0
            row_nodedata.pop(0)
            for nodedata in row_nodedata:
                index_node_map[index] = nodedata
                self.nodedata_map[nodedata] = []
                index += 1

            for node_index, node_line in enumerate(nodedata_file, start=2):
                row_nodedata = self.split(node_line)
                objectid.append(row_nodedata[0])
                row_nodedata.pop(0)
                for i in range(len(row_nodedata)):
                    value = self.safe_float_conversion(row_nodedata[i])
                    self.nodedata_map[index_node_map[i]].append(value)

        # 对 nodedata_map 的每个数组值进行归一化处理
        for key in self.nodedata_map:
            self.nodedata_map[key] = self.normalize_min_max(self.nodedata_map[key])

        if self.noise_type == "attribute":
            noise = np.random.normal(0, self.noise, size=nodenum)
            for key in self.nodedata_map:
                lst = self.nodedata_map[key] + noise
                if min(lst) < 0:
                    lst = lst - min(lst)

                self.nodedata_map[key] = lst
                featuresum_map[key] = max(self.nodedata_map[key])

        similarity_map = self.calculate_similarity(
            self.nodedata_map, featuresum_map, objectid
        )
        self.node_att = similarity_map

        stroke = []
        with open(self.strok_filename, "r", encoding=strok_encoding) as strok_file:
            next(strok_file)
            for stroke_index, stroke_line in enumerate(strok_file, start=2):
                row_strokedata = self.split(stroke_line)
                stroke.append([row_strokedata[1], row_strokedata[2]])

        grouped_data = {}

        for row in stroke:
            key = row[0]
            value = row[1]

            if key not in grouped_data:
                grouped_data[key] = []

            grouped_data[key].append(value)
        self.stroke_map = grouped_data

        probability_matrix = np.zeros((nodenum, nodenum))

        for node_ori, node_out in grouped_data.items():
            outweight_sum = 0
            for out_data in node_out:
                outweight_sum += similarity_map[out_data]
            for out_data in node_out:
                if outweight_sum != 0:
                    value = similarity_map[out_data] / outweight_sum
                else:
                    value = 0
                probability_matrix[int(node_ori) - 1][int(out_data) - 1] = value

            self.ProbabilityMatrix = probability_matrix
        return probability_matrix

    def transform_to_adjacency(self):
        probability_matrix = self.creat_probability_matrix()
        adjacency_matrix = np.where(probability_matrix != 0, 1, probability_matrix)
        self.AdjacencyMatrix = adjacency_matrix
        return adjacency_matrix

    def creat_graph(self, objectid):
        objectid.sort()
        strok_encoding = self.detect_encoding(self.strok_filename)
        nodedata_encoding = self.detect_encoding(self.nodedata_filename)
        index_node_map = {}
        featuresum_map = {}
        graph = nx.DiGraph()

        with open(
            self.nodedata_filename, "r", encoding=nodedata_encoding
        ) as nodedata_file:
            _, row_nodedata = self.get_appointline(nodedata_file, 1)
            index = 0
            row_nodedata.pop(0)
            for nodedata in row_nodedata:
                index_node_map[index] = nodedata
                self.nodedata_map[nodedata] = []
                index += 1

            for node_index, node_line in enumerate(nodedata_file, start=2):

                row_nodedata = self.split(node_line)
                if int(row_nodedata[0]) in objectid:
                    row_nodedata.pop(0)
                    for i in range(len(row_nodedata)):
                        value = self.safe_float_conversion(row_nodedata[i])
                        self.nodedata_map[index_node_map[i]].append(value)
                # else:
                #     self.nodedata_map[index_node_map[i]].append(0)

        # 对 nodedata_map 的每个数组值进行归一化处理
        for key in self.nodedata_map:
            self.nodedata_map[key] = self.normalize_min_max(self.nodedata_map[key])
            featuresum_map[key] = max(self.nodedata_map[key])

        similarity_map = self.calculate_similarity(
            self.nodedata_map, featuresum_map, objectid
        )

        stroke = []
        with open(self.strok_filename, "r", encoding=strok_encoding) as strok_file:
            next(strok_file)
            for stroke_index, stroke_line in enumerate(strok_file, start=2):
                row_strokedata = self.split(stroke_line)
                if (
                    int(row_strokedata[1]) in objectid
                    and int(row_strokedata[2]) in objectid
                ):
                    stroke.append([int(row_strokedata[1]), int(row_strokedata[2])])

        grouped_data = {}

        for row in stroke:
            key = row[0]
            value = row[1]

            if key not in grouped_data:
                grouped_data[key] = []

            grouped_data[key].append(value)

        for node_in, node_out in grouped_data.items():
            outweight_sum = 0
            for out_data in node_out:
                outweight_sum += similarity_map[out_data]
            if outweight_sum == 0:
                continue
            for out_data in node_out:
                value = similarity_map[out_data] / outweight_sum
                if value != 0:
                    graph.add_edge(int(out_data), int(node_in), weight=value)
        return graph

    def creat_graph_1(self):
        strok_encoding = self.detect_encoding(self.strok_filename)
        nodedata_encoding = self.detect_encoding(self.nodedata_filename)
        index_node_map = {}
        featuresum_map = {}
        graph = nx.DiGraph()
        objectid = []

        with open(
            self.nodedata_filename, "r", encoding=nodedata_encoding
        ) as nodedata_file:
            _, row_nodedata = self.get_appointline(nodedata_file, 1)
            index = 0
            row_nodedata.pop(0)
            for nodedata in row_nodedata:
                index_node_map[index] = nodedata
                self.nodedata_map[nodedata] = []
                index += 1

            for node_index, node_line in enumerate(nodedata_file, start=2):

                row_nodedata = self.split(node_line)
                objectid.append(row_nodedata[0])
                row_nodedata.pop(0)
                for i in range(len(row_nodedata)):
                    value = self.safe_float_conversion(row_nodedata[i])
                    self.nodedata_map[index_node_map[i]].append(value)

        # 对 nodedata_map 的每个数组值进行归一化处理
        for key in self.nodedata_map:
            self.nodedata_map[key] = self.normalize_min_max(self.nodedata_map[key])
            featuresum_map[key] = max(self.nodedata_map[key])

        similarity_map = self.calculate_similarity(
            self.nodedata_map, featuresum_map, objectid
        )

        stroke = []
        with open(self.strok_filename, "r", encoding=strok_encoding) as strok_file:
            next(strok_file)
            for stroke_index, stroke_line in enumerate(strok_file, start=2):
                row_strokedata = self.split(stroke_line)
                if (
                    int(row_strokedata[1]) in objectid
                    and int(row_strokedata[2]) in objectid
                ):
                    stroke.append([int(row_strokedata[1]), int(row_strokedata[2])])

        grouped_data = {}

        for row in stroke:
            key = row[0]
            value = row[1]

            if key not in grouped_data:
                grouped_data[key] = []

            grouped_data[key].append(value)

        for node_in, node_out in grouped_data.items():
            outweight_sum = 0
            for out_data in node_out:
                outweight_sum += similarity_map[out_data]
            if outweight_sum == 0:
                continue
            for out_data in node_out:
                value = similarity_map[out_data] / outweight_sum
                if value != 0:
                    graph.add_edge(int(out_data), int(node_in), weight=value)
        return graph

    def probability_to_graph(self):
        graph = nx.DiGraph()
        for row_index, row in enumerate(self.ProbabilityMatrix):
            for col_index, element in enumerate(row):
                if element != 0:
                    weight = element * self.node_att[str(col_index + 1)]
                    graph.add_edge(col_index + 1, row_index + 1, weight=weight)
        return graph

    def adjacency_to_probability_matrix(self, adjacency_matrix):
        n = adjacency_matrix.shape[0]
        adjacency_matrix = adjacency_matrix.transpose()
        for i in range(n):
            d = sum(adjacency_matrix[i])
            if d != 0:
                adjacency_matrix[i] = adjacency_matrix[i] / d
        return adjacency_matrix.transpose()

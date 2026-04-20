import os
import sys
import numpy as np
import chardet

sys.path.append(os.getcwd())
from Tool.JsonTool import read, saveresult, modify
from Tool.MatrixMultiplication import (
    load_result_blocks,
    combine_result_blocks,
)


class ResultProcess:
    def __init__(self, nodedata_txt, result_path, alpha, node_num, noise):
        self.result_path = result_path
        self.alpha = alpha
        self.node_num = node_num
        self.noise = noise
        self.cal_nodenum(nodedata_txt)

    def detect_encoding(self, file_path):
        with open(file_path, "rb") as file:
            return chardet.detect(file.read())["encoding"]

    def cal_nodenum(self, nodedata_txt):
        nodedata_encoding = self.detect_encoding(nodedata_txt)
        with open(nodedata_txt, "r", encoding=nodedata_encoding) as nodedata_file:
            self.node_num = sum(1 for line in nodedata_file) - 1
        if self.node_num <= 800:
            self.block_num = self.node_num * int(self.node_num / 2)
        else:
            self.block_num = 500000
        self.save_node_block_num()

    def save_node_block_num(self):
        configpath = os.getcwd() + "/config.json"
        modify(configpath, self.node_num, self.block_num)

    def read_pagerank(self, result_path):
        result_blocks = load_result_blocks(result_path, group_name="C")
        full_shape = (1, self.node_num)
        result = combine_result_blocks(result_blocks, full_shape)
        return result[0]

    def save_value(self, savepath):
        for alpha in self.alpha:
            result_map = {}
            key_alpha = str(alpha)
            for noise in self.noise:
                result_path = (
                    self.result_path + str(alpha) + "_" + str(int(noise * 100)) + ".h5"
                )
                QPageRank_Weight_result = self.read_pagerank(result_path)
                value = QPageRank_Weight_result
                child_map = {}
                key_sortindex = "sortindex"
                sort_array = np.argsort(value)[::-1]
                sort_array += 1
                sortindex = str(sort_array.tolist())
                child_map[key_sortindex] = sortindex
                key_value = "value"
                child_map[key_value] = str(value.tolist())
                result_map[str(noise)] = child_map
            saveresult(savepath, key_alpha, result_map)


def run_ResultProcess(index):
    configpath = os.getcwd() + "/config.json"
    config = read(configpath)

    nodedata_txt = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/txt/nodedata"
        + str(index)
        + ".txt"
    )
    result_path = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/Matrix/Result/result_alpha_"
    )
    savepath = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/Result/QPageRankWeightResult.json"
    )
    alpha = config["roadselect"]["applyu"]["alpha_s"]
    node_num = config["roadselect"]["global"]["node_num"]
    noise = config["roadselect"]["global"]["noise"]

    resultProcess = ResultProcess(nodedata_txt, result_path, alpha, node_num, noise)
    resultProcess.save_value(savepath)

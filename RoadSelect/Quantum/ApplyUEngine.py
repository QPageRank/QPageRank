import os
import sys
import chardet

sys.path.append(os.getcwd())
from Tool.ApplyU import ApplyU
from Tool.JsonTool import read,modify


class ApplyUEngine:
    def __init__(
        self,
        nodedata_txt,
        state_path,
        start_path,
        u_path,
        output_path,
        max_iterations,
        tolerance,
        alpha_s,
        alphaway,
        noise,
    ):
        self.nodedata_txt=nodedata_txt
        self.state_path = state_path
        self.start_path = start_path
        self.u_path = u_path
        self.output_path = output_path
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.alpha_s = alpha_s
        self.alphaway = alphaway
        self.noise = noise
        self.cal_nodenum(nodedata_txt)
    def detect_encoding(self, file_path):
        with open(file_path, "rb") as file:
            return chardet.detect(file.read())["encoding"]
    def cal_nodenum(self,nodedata_txt):
        nodedata_encoding = self.detect_encoding(nodedata_txt)
        with open(
            nodedata_txt, "r", encoding=nodedata_encoding
        ) as nodedata_file:
            self.node_num = sum(1 for line in nodedata_file) - 1
        if self.node_num <= 800:
            self.block_num = self.node_num * int(self.node_num / 2)
        else:
            self.block_num = 500000
        self.save_node_block_num()

    def save_node_block_num(self):
        configpath = os.getcwd() + "/config.json"
        modify(configpath, self.node_num, self.block_num)

    def run(self, index):
        for noise in self.noise:
            upath = self.u_path + str(int(noise * 100)) + ".h5"
            for alpha in self.alpha_s:
                outputpath = (
                    self.output_path + str(alpha) + "_" + str(int(noise * 100)) + ".h5"
                )
                qPageRankWeight = ApplyU(
                    self.node_num,
                    self.max_iterations,
                    self.tolerance,
                    self.block_num,
                    alpha,
                    self.alphaway,
                )
                qPageRankWeight.apply(
                    upath, self.start_path, self.state_path, outputpath, index
                )


def run_ApplyUEngine(index):
    configpath = os.getcwd() + "/config.json"
    config = read(configpath)
    node_num = config["roadselect"]["global"]["node_num"]
    nodedata_txt = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/txt/nodedata"
        + str(index)
        + ".txt"
    )
    state_path = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/Temp/nextstate"
    )
    start_path = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/Matrix/start.h5"
    )
    u_path = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/Matrix/U_"
    )
    output_path = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/Matrix/Result/result_alpha_"
    )
    max_iterations = config["roadselect"]["applyu"]["max_iterations"]
    tolerance = config["roadselect"]["applyu"]["tolerance"]
    alphaway = config["roadselect"]["applyu"]["alphaway"]
    alpha_s = config["roadselect"]["applyu"]["alpha_s"]
    block_num = config["roadselect"]["global"]["block_num"]
    noise = config["roadselect"]["global"]["noise"]
    applyUEngine = ApplyUEngine(
        nodedata_txt,
        state_path,
        start_path,
        u_path,
        output_path,
        max_iterations,
        tolerance,
        alpha_s,
        alphaway,
        noise,
    )
    applyUEngine.run(index)

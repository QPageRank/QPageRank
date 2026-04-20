import numpy as np
import numpy as np
import os
import sys

sys.path.append(os.getcwd())
from Tool.CreatStart import CreatStart
from RoadSelect.Quantum.CreatProbabilityMatrix import CreatProbabilityMatrix
from Tool.JsonTool import read


class CreatStartEngine:
    def __init__(
        self,
        start_output_hdf5_filename,
        strok_filename,
        nodedata_filename,
        block_size,
        alpha,
        block_num,
    ):
        self.start_output_hdf5_filename = start_output_hdf5_filename
        self.strok_filename = strok_filename
        self.nodedata_filename = nodedata_filename
        self.block_size = block_size
        self.alpha = alpha
        self.block_num = block_num

    def run(self):
        creatProbabilityMatrix = CreatProbabilityMatrix(
            self.strok_filename, self.nodedata_filename
        )
        probabilityMatrix = creatProbabilityMatrix.ProbabilityMatrix
        node_att = creatProbabilityMatrix.node_att
        node_att = {key: 1 for key in node_att}
        self.creat_start(
            probabilityMatrix, node_att, self.start_output_hdf5_filename, self.block_num
        )

    def creat_start(
        self, prob_matrix_start, node_att, start_output_hdf5_filename, block_num
    ):
        # 确保 probability_matrix 的数据类型为 np.float32
        prob_matrix_start = prob_matrix_start.astype(np.float32)

        # 初始化 QPageRankWeight
        creatstart = CreatStart(
            prob_matrix_start, node_att, self.alpha, self.block_size, block_num
        )
        creatstart.creat_start(start_output_hdf5_filename)


def run_CreatStartEngine(index):
    configpath = os.getcwd() + "/config.json"
    config = read(configpath)

    stroke_txt = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/txt/stroke"
        + str(index)
        + ".txt"
    )
    nodedata_txt = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/txt/nodedata"
        + str(index)
        + ".txt"
    )
    start_output_hdf5_filename = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/Matrix/start.h5"
    )
    block_size = config["roadselect"]["creatstart"]["block_size"]
    alpha = config["roadselect"]["creatstart"]["alpha"]
    block_num = config["roadselect"]["global"]["block_num"]
    creatStartEngine = CreatStartEngine(
        start_output_hdf5_filename,
        stroke_txt,
        nodedata_txt,
        block_size,
        alpha,
        block_num,
    )
    creatStartEngine.run()

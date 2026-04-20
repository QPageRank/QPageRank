import numpy as np
import os
import sys

sys.path.append(os.getcwd())
from Tool.CreatU import CreatU
from RoadSelect.Quantum.CreatProbabilityMatrix import CreatProbabilityMatrix
from Tool.JsonTool import read


class CreatUEngine:
    def __init__(
        self,
        strok_filename,
        nodedata_filename,
        proj_output_hdf5_filename,
        swap_output_hdf5_filename,
        u_output_hdf5_filename,
        temp_path,
        max_cache_blocks,
        block_num,
        noise,
        evolution_way,
        alpha,
    ):
        self.strok_filename = strok_filename
        self.nodedata_filename = nodedata_filename
        self.proj_output_hdf5_filename = proj_output_hdf5_filename
        self.swap_output_hdf5_filename = swap_output_hdf5_filename
        self.u_output_hdf5_filename = u_output_hdf5_filename
        self.temp_path = temp_path
        self.max_cache_blocks = max_cache_blocks
        self.block_num = block_num
        self.noise = noise
        self.evolution_way = evolution_way
        self.alpha = alpha

    def run(self):
        for noise in self.noise:
            output_hdf5_filename = (
                self.u_output_hdf5_filename + str(int(noise * 100)) + ".h5"
            )
            creatProbabilityMatrix = CreatProbabilityMatrix(
                self.strok_filename, self.nodedata_filename, "attribute", noise
            )
            probabilityMatrix = creatProbabilityMatrix.ProbabilityMatrix
            if self.evolution_way == "1":
                n = probabilityMatrix.shape[1]
                ones_matrix = np.ones_like(probabilityMatrix) * (1 / n)
                probabilityMatrix = (
                    self.alpha * probabilityMatrix + (1 - self.alpha) * ones_matrix
                )
            self.creatU(
                probabilityMatrix,
                self.proj_output_hdf5_filename,
                self.swap_output_hdf5_filename,
                output_hdf5_filename,
                self.temp_path,
                self.block_num,
            )

    def creatU(
        self,
        prob_matrix,
        proj_output_hdf5_filename,
        swap_output_hdf5_filename,
        u_output_hdf5_filename,
        temp_path,
        block_num,
    ):
        # 确保 probability_matrix 的数据类型为 np.float32
        probability_matrix = prob_matrix.astype(np.float32)

        # 初始化 QPageRankWeight
        qPageRankWeight = CreatU(
            probability_matrix, temp_path, self.max_cache_blocks, block_num
        )
        qPageRankWeight.creatU(
            proj_output_hdf5_filename, swap_output_hdf5_filename, u_output_hdf5_filename
        )


def run_CreatUEngine(index):
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
    proj_output_hdf5_filename = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/Temp/proj.h5"
    )
    swap_output_hdf5_filename = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/Temp/swap.h5"
    )
    u_output_hdf5_filename = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/Matrix/U_"
    )
    temp_path = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/Temp/"
    )
    max_cache_blocks = config["roadselect"]["creatu"]["max_cache_blocks"]
    block_num = config["roadselect"]["global"]["block_num"]
    noise = config["roadselect"]["global"]["noise"]
    evolution_way = config["roadselect"]["global"]["evolution_way"]
    alpha = config["roadselect"]["creatu"]["alpha"]
    creatUEngine = CreatUEngine(
        stroke_txt,
        nodedata_txt,
        proj_output_hdf5_filename,
        swap_output_hdf5_filename,
        u_output_hdf5_filename,
        temp_path,
        max_cache_blocks,
        block_num,
        noise,
        evolution_way,
        alpha,
    )
    creatUEngine.run()

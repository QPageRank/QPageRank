import os
import sys
import numpy as np

sys.path.append(os.getcwd())
from RoadSelect.Quantum.CreatProbabilityMatrix import CreatProbabilityMatrix
from Tool.JsonTool import read, saveresult, save
from RoadSelect.Classic.PageRankWeight import PageRankWeight


class PageRankWeightEngine:
    def __init__(
        self,
        strok_filename,
        nodedata_filename,
        max_iterations,
        tolerance,
        alpha,
        resultpath,
        noise,
        diffpath,
    ):
        self.strok_filename = strok_filename
        self.nodedata_filename = nodedata_filename
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.alpha = alpha
        self.resultpath = resultpath
        self.noise = noise
        self.diffpath = diffpath

    def run(self):
        for alpha in self.alpha:
            child_map_1 = {}
            key_alpha = str(alpha)
            for noise in self.noise:
                child_map_2 = {}
                creatProbabilityMatrix = CreatProbabilityMatrix(
                    self.strok_filename, self.nodedata_filename, "attribute", noise
                )
                probabilityMatrix = creatProbabilityMatrix.ProbabilityMatrix
                node_att = creatProbabilityMatrix.node_att
                pageRankWeight = PageRankWeight(
                    probabilityMatrix, self.max_iterations, node_att
                )
                result_pageRankWeight, diff_list = pageRankWeight.page_rank(alpha)
                PageRank_Weight_result = np.array(result_pageRankWeight)
                sort_array = np.argsort(PageRank_Weight_result)[::-1]
                sort_array += 1
                sortindex = str(sort_array.tolist())
                key_sortindex = "sortindex"
                child_map_2[key_sortindex] = sortindex
                key_value = "value"
                child_map_2[key_value] = str(PageRank_Weight_result.tolist())
                child_map_1[str(noise)] = child_map_2
            saveresult(self.resultpath, key_alpha, child_map_1)
            diff_map = {}
            diff_map["value"] = str(diff_list)
            # save(self.diffpath, diff_map)


def run_ClassicEngine(index):
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
    max_iterations = config["roadselect"]["classic"]["max_iterations"]
    tolerance = config["roadselect"]["classic"]["tolerance"]
    alpha = config["roadselect"]["classic"]["alpha"]
    resultpath = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/Result/PageRankWeightResult.json"
    )
    noise = config["roadselect"]["global"]["noise"]
    diffpath = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/Result/class_diff_result.json"
    )

    pageRankWeightEngine = PageRankWeightEngine(
        stroke_txt,
        nodedata_txt,
        max_iterations,
        tolerance,
        alpha,
        resultpath,
        noise,
        diffpath,
    )
    pageRankWeightEngine.run()

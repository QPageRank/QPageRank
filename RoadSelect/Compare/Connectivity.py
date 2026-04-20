import os
import sys
import json
import numpy as np
import random
import networkx as nx
from tqdm import tqdm

sys.path.append(os.getcwd())
from RoadSelect.Quantum.CreatProbabilityMatrix import CreatProbabilityMatrix
from Tool.JsonTool import read, save


def calculate_path_efficiency_change(G_original, G_sub, N=1000, weight=None):

    orig_dists = dict(nx.all_pairs_dijkstra_path_length(G_original, weight=weight))
    sub_dists = dict(nx.all_pairs_dijkstra_path_length(G_sub, weight=weight))
    nodes = list(G_original.nodes())
    ratios = []
    attempts = 0
    while len(ratios) < N and attempts < N * 10:
        i, j = random.sample(nodes, 2)
        d_ij = orig_dists.get(i, {}).get(j)
        dp_ij = sub_dists.get(i, {}).get(j)
        if d_ij and dp_ij:
            ratios.append(d_ij / dp_ij)
        attempts += 1
    return float(np.mean(ratios)) if ratios else float("inf")


def run(strok_filename, nodedata_filename, select_result_path, connectivity_path):
    result_map = {}
    cpm = CreatProbabilityMatrix(strok_filename, nodedata_filename)
    G_orig = cpm.probability_to_graph()
    select_result = read(select_result_path)

    for key1, v1 in tqdm(select_result.items(), desc="Level1"):
        map1 = {}
        for key2, v2 in tqdm(
            v1.get("0.4", {}).items(), desc=f"{key1} Level2", leave=False
        ):
            map2 = {}
            for key3, v3 in tqdm(v2.items(), desc=f"{key1}-{key2} Level3", leave=False):
                objectid = json.loads(v3)
                G_sub = cpm.creat_graph(objectid)
                map2[key3] = {
                    "efficiency": calculate_path_efficiency_change(
                        G_orig, G_sub, weight="weight"
                    )
                }
            map1[key2] = map2
        result_map[key1] = map1

    save(connectivity_path, result_map)


def run_R_path(index):
    cfg = read(os.getcwd() + "/config.json")
    stroke_txt = (
        os.getcwd()
        + cfg["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/txt/stroke"
        + str(index)
        + ".txt"
    )
    nodedata_txt = (
        os.getcwd()
        + cfg["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/txt/nodedata"
        + str(index)
        + ".txt"
    )
    selectresult_path = (
        os.getcwd()
        + cfg["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/Result/selectresult.json"
    )
    connectivity_path = (
        os.getcwd()
        + cfg["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/Result/connectivity.json"
    )
    run(
        stroke_txt,
        nodedata_txt,
        selectresult_path,
        connectivity_path,
    )


if __name__ == "__main__":
    index = 116
    run_R_path(index)

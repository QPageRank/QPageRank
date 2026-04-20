import os
import sys
import geopandas as gpd
import fiona
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, BatchNorm
from tqdm import tqdm

sys.path.append(os.getcwd())
from Tool.JsonTool import read, save


# 读取配置，获取属性列表等全局设置
def load_config(index):
    cfg = read(os.getcwd() + "/config.json")
    gnn_cfg = cfg["roadselect"]["GNN"]
    global TRAIN_SHP, TEST_SHP, OUT_PATH, NOISES, ATTRIBUTES
    TRAIN_SHP = os.getcwd() + "/Data/RoadSelect/GNN/guangzhou_road_stroke.shp"
    TEST_SHP = "/Data/RoadSelect/" + str(index) + "/Roadgdbdata/" + str(index) + ".gdb"
    OUT_PATH = os.getcwd() + "/Data/RoadSelect/" + str(index) + "/Result/GNNResult.json"
    NOISES = cfg["roadselect"]["global"]["noise"]
    ATTRIBUTES = gnn_cfg.get("attributes", ["fclass", "Length"])
    return cfg


def load_graph(file_path, file_format="shp", layer=None):
    """
    构建对偶图：节点为道路要素(FID)，边表示道路要素在空间网络中相邻（共享端点）。
    """
    # 读取原始道路要素
    if file_format == "shp":
        gdf = gpd.read_file(file_path)
    elif file_format == "gdb":
        gpkg = os.getcwd() + file_path
        layers = fiona.listlayers(gpkg) if layer is None else [layer]
        gdf = gpd.read_file(gpkg, layer=layers[0])
    else:
        raise ValueError(f"Unsupported format: {file_format}")
    # 构建底层点-路网映射
    node_to_fids = {}
    for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Mapping nodes to roads"):
        geom = row.geometry
        # 使用 gdf.index 作为 ID 当格式为 gdb，否则使用指定字段
        fid = row.name
        lines = (
            [geom]
            if geom.geom_type == "LineString"
            else geom.geoms if geom.geom_type == "MultiLineString" else []
        )
        for line in lines:
            coords = list(line.coords)
            for pt in coords:
                node_to_fids.setdefault(pt, set()).add(fid)
        # 构建对偶图：节点为 fid
    DG = nx.Graph()
    # 获取道路要素 ID 列表
    fid_list = gdf.index.tolist()
    for fid in fid_list:
        DG.add_node(fid)
    # 添加属性
    for idx, row in gdf.iterrows():
        # 取 fid
        fid = idx
        for k in ATTRIBUTES:
            DG.nodes[fid][k] = row.get(k, 0)
        DG.nodes[fid]["label_1"] = row.get("label_1", 0)
    # 添加对偶图边：任意在底层图共享节点的道路要素
    for fids in node_to_fids.values():
        flist = list(fids)
        for i in range(len(flist)):
            for j in range(i + 1, len(flist)):
                DG.add_edge(flist[i], flist[j])
    return DG


def generate_data(G, split=(0.7, 0.15, 0.15)):
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    # 动态构建特征矩阵
    x = [[float(G.nodes[n].get(k, 0)) for k in ATTRIBUTES] for n in node_list]
    y = [G.nodes[n].get("label_1", 0) for n in node_list]
    # 对偶图节点即为道路 fid，直接使用 node_list
    fids = node_list.copy()
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    edges = [[node_to_idx[u], node_to_idx[v]] for u, v in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index, y=y)
    n = data.num_nodes
    perm = torch.randperm(n)
    t = int(split[0] * n)
    v = t + int(split[1] * n)
    data.train_mask = torch.zeros(n, dtype=torch.bool)
    data.val_mask = torch.zeros(n, dtype=torch.bool)
    data.test_mask = torch.zeros(n, dtype=torch.bool)
    data.train_mask[perm[:t]] = True
    data.val_mask[perm[t:v]] = True
    data.test_mask[perm[v:]] = True
    return data, node_list, fids


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x).relu()
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    return out


def runGNN(index):
    # 加载配置和全局变量
    cfg = load_config(index)
    TRAIN_FORMAT = cfg["roadselect"]["GNN"].get("train_format", "shp")
    TEST_FORMAT = cfg["roadselect"]["GNN"].get("test_format", "shp")
    TRAIN_LAYER = cfg["roadselect"]["GNN"].get("train_layer", None)
    TEST_LAYER = "road_" + str(index)
    # TEST_LAYER = cfg["roadselect"]["GNN"].get("test_layer", None)
    # 训练阶段
    G_train = load_graph(TRAIN_SHP, file_format=TRAIN_FORMAT, layer=TRAIN_LAYER)
    data_train, _, _ = generate_data(G_train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphSAGE(
        in_channels=data_train.num_node_features,
        hidden_channels=64,
        out_channels=int(data_train.y.max().item()) + 1,
    ).to(device)
    data_train = data_train.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=50)
    best, wait = 1e9, 0
    for epoch in range(1, 1001):
        loss = train(model, data_train, opt)
        outv = evaluate(model, data_train)
        valloss = F.nll_loss(
            outv[data_train.val_mask], data_train.y[data_train.val_mask]
        )
        sched.step(valloss)
        # 打印每次训练信息
        print(f"Epoch {epoch:04d} | Train Loss: {loss:.4f} | Val Loss: {valloss:.4f}")
        if valloss < best:
            best, wait = valloss, 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            wait += 1
            if wait >= 100:
                print("Early stopping.")
                break

    # 测试准备
    G_test = load_graph(TEST_SHP, file_format=TEST_FORMAT, layer=TEST_LAYER)
    data_test, node_list, fids = generate_data(G_test)
    data_test = data_test.to(device)
    model.load_state_dict(torch.load("best_model.pth"))

    gnn_map = {}
    for noise in tqdm(NOISES, desc="Processing noise levels"):
        data_test.x = data_test.x * (1 + noise * torch.randn_like(data_test.x))
        out = evaluate(model, data_test)
        probs = torch.exp(out[:, 1]).cpu().numpy()
        # 聚合节点分值到道路级别：取每条路(fid)的最大节点分值
        fid_scores = {}
        for i, fid in enumerate(fids):
            fid_scores.setdefault(fid, []).append(probs[i])
        agg_scores = {fid: max(scores) for fid, scores in fid_scores.items()}
        # 按聚合分值排序
        sorted_items = sorted(
            agg_scores.items(), key=lambda item: item[1], reverse=True
        )
        sorted_f = [fid + 1 for fid, score in sorted_items]
        gnn_map[str(noise)] = {"sorted": str(sorted_f)}

    # 保存
    save(OUT_PATH, gnn_map)


if __name__ == "__main__":
    index = 1
    runGNN(index)

import os
import sys
import subprocess

sys.path.append(os.getcwd())
from Tool.JsonTool import read

from Preprocessing.stroke.StrokeEngine import run_stroke
from RoadSelect.Quantum.CreatStartEngine import run_CreatStartEngine
from RoadSelect.Quantum.CreatUEngine import run_CreatUEngine
from RoadSelect.Quantum.ApplyUEngine import run_ApplyUEngine
from RoadSelect.Quantum.ResultProcess import run_ResultProcess

from RoadSelect.Classic.PageRankWeightEngine import run_ClassicEngine

from RoadSelect.GNN.GNN import runGNN

from RoadSelect.Compare.SelectRate import run_SelectRate
from RoadSelect.Compare.RoadMatch import run_RoadMatch
from RoadSelect.Compare.Connectivity import run_R_path
from RoadSelect.Compare.Caldensity import run_density

import logging

configpath = os.getcwd() + "/config.json"
config = read(configpath)

python2_path = config["python2path"]
shptogdb_path = os.getcwd() + "/Tool/ShpToGDB.py"
neartable_path = os.getcwd() + "/Tool/NearTable.py"

# 配置日志：写入到 debug.log，级别 DEBUG（会捕获 INFO/WARNING/ERROR）
log_dir = os.getcwd() + config["roadselect"]["preprocessing"]["basicpath"] + "/Log"

log_file = os.path.join(log_dir, "evolutionary_debug.log")

# 2. 如果日志目录不存在，就创建它
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=log_file,
    filemode="a",  # 'w' 每次重写，'a' 则追加
    level=logging.DEBUG,  # 捕获 DEBUG 及以上级别日志
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    for index in range(113, 114):
        index = 2
        # run_stroke(index)
        # subprocess.check_call([python2_path, shptogdb_path, str(index)])
        # subprocess.check_call([python2_path, neartable_path, str(index)])

        # # QPageRank
        # run_CreatStartEngine(index)
        # run_CreatUEngine(index)
        # run_ApplyUEngine(index)
        # run_ResultProcess(index)

        # # CPageRank
        # run_ClassicEngine(index)

        # GNN
        runGNN(index)

        run_SelectRate(index)
        # run_RoadMatch(index)
        # run_R_path(index)
        # run_density(index)


if __name__ == "__main__":
    main()

import numpy as np
import os
import math
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from scipy.sparse import csr_matrix
from concurrent.futures import ThreadPoolExecutor, as_completed
from Tool.JsonTool import modify
from Tool.MatrixMultiplication import (
    BlockSparseMatrix,
)


class CreatStart:
    def __init__(self, prob_matrix_start, node_att, alpha, block_size, block_num):
        self.alpha = alpha
        self.prob_matrix_start = prob_matrix_start
        self.node_att = node_att
        self.node_num = prob_matrix_start.shape[0]
        if self.node_num <= 800:
            self.block_num = self.node_num * int(self.node_num / 2)
        else:
            self.block_num = 500000
        self.block_size = block_size
        self.save_node_block_num()

    def save_node_block_num(self):
        configpath = os.getcwd() + "/config.json"
        modify(configpath, self.node_num, self.block_num)

    # 将阻尼系数加入prob_matrix_start
    def add_alpha_to_prob_matrix_start(self):
        matrix = self.prob_matrix_start * self.alpha + (1 - self.alpha) / self.node_num
        self.prob_matrix_start = matrix

    def create_start_matrix_multithread_block(self, max_workers=8):
        """
        分块处理，并添加全局进度条：每处理完一块的列数据，就更新全局 start_matrix，
        同时通过进度条展示已处理的总列数。
        """
        total_size = self.node_num * self.node_num
        sqrt_prob = np.sqrt(self.prob_matrix_start)

        # 全局初态矩阵
        start_matrix = csr_matrix((total_size, 1), dtype=np.float32)

        # 全局进度条，追踪已处理的列数
        pbar = tqdm(total=self.node_num, desc="Processing columns")

        def process_col(col):
            # 计算当前列对应的行索引 a-a,a-b,a-c,a-d
            row_indices = np.arange(self.node_num) + col * self.node_num
            amplitudes = sqrt_prob[col] * math.sqrt(self.node_att[str(col + 1)])
            # 构造当前列的稀疏向量
            vec = csr_matrix(
                (amplitudes, (row_indices, np.zeros(self.node_num, dtype=int))),
                shape=(total_size, 1),
                dtype=np.float32,
            )
            return vec

        # 按块处理，每个块处理 block_size 列
        for block_start in range(0, self.node_num, self.block_size):
            block_end = min(block_start + self.block_size, self.node_num)
            partial_results = []  # 存储当前块的中间结果
            # max_workers=max_workers
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(process_col, col): col
                    for col in range(block_start, block_end)
                }
                for future in as_completed(futures):
                    partial_results.append(future.result())
                    # 每完成一个col的计算，更新进度条
                    pbar.update(1)

            # 累加当前块内的所有结果到全局矩阵
            block_matrix = sum(
                partial_results, csr_matrix((total_size, 1), dtype=np.float32)
            )
            start_matrix += block_matrix

            # 释放当前块的中间数据
            del partial_results, block_matrix

        pbar.close()
        # 全局归一化：计算振幅平方和 S，然后除以 sqrt(S)
        # 注意 sparse 矩阵 .power 或 .multiply 也行，但这里转换为 dense 便于示例
        amplitudes_dense = start_matrix.toarray().ravel()
        S = float(np.sum(np.abs(amplitudes_dense) ** 2))
        norm_factor = 1.0 / math.sqrt(S)
        start_matrix *= norm_factor
        return start_matrix

    def creat_start(self, start_output_hdf5_filename):
        # 构造初态
        start = self.create_start_matrix_multithread_block()
        start_block = BlockSparseMatrix.from_matrix(start, (self.block_num, 1))
        start_block.write_to_h5(
            start_output_hdf5_filename, group_name="C", max_cache_blocks=5
        )

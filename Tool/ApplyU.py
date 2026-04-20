import numpy as np
import math
import json
import os
from scipy.stats import kendalltau
import logging
from scipy.sparse import csr_matrix
from Tool.MatrixMultiplication import (
    BlockSparseMatrix,
    H5BlockMultiplicationEngine,
    get_start_state,
)
from Tool.JsonTool import saveresult

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ApplyU:
    def __init__(self, node_num, max_iterations, tolerance, block_num, alpha, alphaway):
        self.node_num = node_num
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.alpha = alpha
        self.alphaway = alphaway
        self.block_num = block_num

    # 初态拆分，测量，将初始态进行分割，分割为两部分，一部分为待测量，一部分为剩下的
    def separate_base(self, state):
        # 将 state 转换为 NumPy 数组（如果它还不是）
        # state_np = state.toarray()

        # 创建行和列的索引
        index_1, index_2 = np.divmod(
            np.arange(self.node_num * self.node_num), self.node_num
        )

        # 将 state 的值与索引组合
        result = np.column_stack((state.flatten(), index_1, index_2))

        # 转换为列表（如果需要）
        return result.tolist()

    # 分离演化策略
    def separatist_evolution(self, state, alpha):
        ones_matrix = np.ones_like(state) / math.sqrt(len(state))
        nextstate = math.sqrt(alpha) * state + math.sqrt(1 - alpha) * ones_matrix
        return nextstate

    # u_operator_op的迭代次数确认
    def evolutionary_state(
        self, u_operator, start_path, state_path, output_path, index
    ):
        iteration = 0
        max_i = self.max_iterations
        result = []
        result_noseparate = []
        result_parameter = {}
        next_state = get_start_state(start_path, self.node_num * self.node_num, 1)
        result_noseparate.append(next_state.ravel())
        if self.alphaway == "1":
            separatist_evolution_result = self.separatist_evolution(
                next_state, self.alpha
            )
            result_1 = self.separate_base(separatist_evolution_result)
        else:
            result_1 = self.separate_base(next_state)
        next_result = self.celiang(result_1)
        result.append(next_result)
        diff_map = {}
        check = True
        while iteration < max_i and check:
            pre_path_1 = state_path + "_" + str(iteration) + ".h5"
            iteration += 1
            next_path_1 = state_path + "_" + str(iteration) + ".h5"
            progress_desc = "第" + str(iteration) + "次迭代"
            if iteration == 1:
                next_state = self.apply_u_operator(
                    u_operator, start_path, next_path_1, "第1次迭代"
                )
                result_noseparate.append(next_state.ravel())
            else:
                next_state = self.apply_u_operator(
                    u_operator, pre_path_1, next_path_1, progress_desc
                )
                result_noseparate.append(next_state.ravel())

            if self.alphaway == "1":
                separatist_evolution_result = self.separatist_evolution(
                    next_state, self.alpha
                )
                result_1 = self.separate_base(separatist_evolution_result)
            else:
                result_1 = self.separate_base(next_state)
            next_result = self.celiang(result_1)
            result.append(next_result)
            x_curr = np.mean(result, axis=0)
            diff_1, angle_rad_1, diff_2, angle_rad_2 = self.should_stop_iteration(
                result_noseparate
            )

            print(
                "第"
                + str(iteration)
                + "次迭代变化 求平均后模长："
                + str(diff_1)
                + " 求平均后角度："
                + str(angle_rad_1)
                + "次迭代变化 模长："
                + str(diff_2)
                + " 角度："
                + str(angle_rad_2)
                # + " 计算得到差值："
                # + str(result)
            )
            diff_child_list = []
            diff_child_list.append(diff_1)
            diff_child_list.append(angle_rad_1)
            diff_child_list.append(diff_2)
            diff_child_list.append(angle_rad_2)

            diff_path = (
                os.getcwd()
                + "/Data/RoadSelect/"
                + str(index)
                + "/Result/result_diff_"
                + str(self.alpha)
                + ".json"
            )
            # saveresult(diff_path, str(iteration), str(diff_child_list))
            remove_path = state_path + "_" + str(iteration - 2) + ".h5"
            self.remove(remove_path)
        remove_path = state_path + "_" + str(iteration) + ".h5"
        self.remove(remove_path)
        remove_path = state_path + "_" + str(iteration - 1) + ".h5"
        self.remove(remove_path)
        next_result_block = BlockSparseMatrix.from_matrix(
            csr_matrix(x_curr), (1, self.node_num)
        )
        next_result_block.write_to_h5(output_path, group_name="C", max_cache_blocks=20)
        serializable_result = [arr.tolist() for arr in result]
        if index != -1:
            out_path = (
                os.getcwd() + "/Data/RoadSelect/" + str(index) + "/Result/result.json"
            )
            # saveresult(out_path, self.alpha, serializable_result)
            out_path_parameter = (
                os.getcwd()
                + "/Data/RoadSelect/"
                + str(index)
                + "/Result/result_parameter.json"
            )
            # saveresult(out_path_parameter, self.alpha, result_parameter)
        return x_curr, iteration

    # 应用演化算符而不显式构造u_operator
    def apply_u_operator(self, u_operator, previous_state, next_path, progress_desc):

        total_size = self.node_num * self.node_num
        engine = H5BlockMultiplicationEngine(
            u_operator,
            previous_state,
            block_shape_A=(self.block_num, self.block_num),
            block_shape_B=(self.block_num, 1),
            full_shape_A=(total_size, total_size),
            full_shape_B=(total_size, 1),
            output_hdf5_filename=next_path,
            max_cache_blocks=100,
            progress_desc=progress_desc,
        )
        logging.debug("本次酉算符迭代的输出路径" + next_path)
        engine.multiply()
        next_state = engine.get_combined_result()

        return next_state

    def calculate_element_wise_average_difference(self, arrays):
        # 计算前n-1次迭代结果中每个元素的平均值
        avg_array_n_minus_1 = np.mean(arrays[:-1], axis=0)

        # 计算前n次迭代结果中每个元素的平均值
        avg_array_n = np.mean(arrays, axis=0)

        # 计算这两个平均值数组之间的差值的绝对值
        absolute_differences = np.abs(avg_array_n - avg_array_n_minus_1)

        # 计算这些绝对值的平均值
        average_of_absolute_differences = np.mean(absolute_differences)

        norm = np.sum(avg_array_n)
        avg_array_n = avg_array_n / norm
        dic = []
        for i in range(self.node_num):
            dic.append(np.around(avg_array_n[i].real, 5))

        return dic, average_of_absolute_differences

    def should_stop_iteration(
        self,
        arrays,
    ):
        x_prev = np.mean(arrays[:-1], axis=0)

        x_curr = np.mean(arrays, axis=0)

        # 1、求平均状态相减向量模长
        diff_1 = np.linalg.norm(x_curr - x_prev)
        # 2、计算平均状态的夹角
        norm1 = np.linalg.norm(x_curr)
        norm2 = np.linalg.norm(x_prev)
        dot_product = np.dot(x_curr, x_prev)
        cos_theta = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)
        angle_rad_1 = np.rad2deg(angle_rad)

        # 3、全局状态相减向量模长
        pre = arrays[-1]
        curr = arrays[-2]
        diff_2 = np.linalg.norm(curr - pre)

        # 4、计算全局状态的夹角
        norm1 = np.linalg.norm(curr)
        norm2 = np.linalg.norm(pre)
        dot_product = np.dot(curr, pre)
        cos_theta = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)
        angle_rad_2 = np.rad2deg(angle_rad)

        return diff_1, angle_rad_1, diff_2, angle_rad_2

    def celiang1(self, result):
        qpagerank = []
        for index in range(self.node_num):
            tem = sum(child[0] ** 2 for child in result if child[2] == index)
            qpagerank.append(tem)
        return np.array(qpagerank)

    def celiang(self, result):
        # 按照 child[2] 的值对 result 进行分组
        grouped_result = {}
        for child in result:
            index = child[2]
            if index not in grouped_result:
                grouped_result[index] = []
            grouped_result[index].append(child[0])
        # 计算每个 index 的平方和
        qpagerank = []
        for index in range(self.node_num):
            values = grouped_result.get(index, [])
            qpagerank.append(np.sum(np.array(values) ** 2))
        return np.array(qpagerank)

    def apply(self, u_operator, start_path, state_path, output_path, index=-1):
        # 执行演化状态
        average, iteration = self.evolutionary_state(
            u_operator, start_path, state_path, output_path, index
        )

        return (average, iteration)

    def remove(self, file_path):
        try:
            os.remove(file_path)
            print(f"文件 {file_path} 已删除。")
        except FileNotFoundError:
            print(f"文件 {file_path} 不存在。")
        except PermissionError:
            print(f"没有权限删除文件 {file_path}。")
        except Exception as e:
            print(f"删除文件时出错: {e}")


# next_path = "PageRank/Tool/UMatrix/temp/nextstate.h5"
# pre_path = "PageRank/Tool/UMatrix/temp/prestate.h5"
# start_path = "PageRank/Tool/UMatrix/start"
# u_path = "PageRank/Tool/UMatrix/U"
# entries = os.listdir(start_path)
# max_iterations = 100
# tolerance = 0.000000001
# for i in range(len(entries)):
#     start = start_path + "/start_" + str(i) + ".h5"
#     u_operator = u_path + "/u_" + str(i) + ".h5"
#     output_path = "PageRank/Tool/UMatrix/result/result_" + str(i) + ".h5"
#     qPageRankWeight = ApplyU(4096, max_iterations, tolerance)
#     qPageRankWeight.apply(u_operator, start, pre_path, next_path, output_path)
# a = 1

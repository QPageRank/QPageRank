import numpy as np
import os
import math
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import expm
from Tool.MatrixMultiplication import (
    SparseMultiplicationEngine,
    BlockSparseMatrix,
    H5BlockMultiplicationEngine,
    H5BlockSubtractionEngine,
    H5BlockScalarMultiplicationEngine,
    H5BlockComplexMultiplyEngine,
)


class CreatU:
    def __init__(self, prob_matrix, temp_path, max_cache_blocks, block_num):
        self.prob_matrix = prob_matrix
        self.node_num = prob_matrix.shape[0]
        self.temp_path = temp_path
        self.max_cache_blocks = max_cache_blocks
        self.block_num = block_num

    def create_proj_matrix_vectorized(self, proj_output_hdf5_filename):
        n = self.node_num
        total_size = n * n
        rows = np.arange(total_size)
        cols = np.tile(np.arange(n), n)
        # 将矩阵转换为 float32 后计算平方根，并按 Fortran 顺序展平以匹配原来双重循环的数据顺序
        data = np.sqrt(self.prob_matrix.astype(np.float32)).ravel("F")
        R = csr_matrix((data, (rows, cols)), shape=(total_size, n), dtype=np.float32)

        # 一次性计算 R @ R.T
        engine = SparseMultiplicationEngine(
            R,
            R.T,
            block_shape_A=(self.block_num, n),
            block_shape_B=(n, self.block_num),
            hdf5_filename=proj_output_hdf5_filename,
            max_cache_blocks=self.max_cache_blocks,
            progress_desc="投影算符计算",
        )
        engine.run()

    # 生成交换算符
    def create_swap_matrix(self, swap_output_hdf5_filename):
        N = self.node_num
        total_size = N * N

        # 生成所有行和列的网格
        row, col = np.mgrid[0:N, 0:N]

        # 计算展平后的索引
        index_r = (row + col * N).ravel()
        index_l = (col + row * N).ravel()

        # 创建COO矩阵并转换为CSR格式
        swap = coo_matrix(
            (np.ones(total_size), (index_r, index_l)),
            shape=(total_size, total_size),
            dtype=np.float32,
        ).tocsr()

        swap_block = BlockSparseMatrix.from_matrix(
            swap, (self.block_num, self.block_num)
        )
        swap_block.write_to_h5(
            swap_output_hdf5_filename,
            group_name="C",
            max_cache_blocks=self.max_cache_blocks,
        )

    def creat_u_operator_op(self, proj, swap, u_output_hdf5_filename):
        # U = (2SProjS - I)(2Proj - I)
        N = self.node_num
        total_size = N * N

        # 现在使用 H5BlockScalarMultiplicationEngine 对该文件中的矩阵乘以一个常数
        scalar = 2.0  # 例如，将矩阵乘以 2
        engine = H5BlockScalarMultiplicationEngine(
            proj,
            self.temp_path + "M1.h5",
            scalar,
            full_shape=(total_size, total_size),
            block_shape=(self.block_num, self.block_num),
            group_name="C",
            max_cache_blocks=self.max_cache_blocks,
            progress_desc="M1=2*投影",
        )
        engine.multiply_scalar()

        engine1 = H5BlockMultiplicationEngine(
            swap,
            self.temp_path + "M1.h5",
            block_shape_A=(self.block_num, self.block_num),
            block_shape_B=(self.block_num, self.block_num),
            full_shape_A=(total_size, total_size),
            full_shape_B=(total_size, total_size),
            output_hdf5_filename=self.temp_path + "M2.h5",
            max_cache_blocks=self.max_cache_blocks,
            progress_desc="计算 M2 = swap@M1",
        )
        engine1.multiply()

        # Step 2: 计算 M = M1@A, 注意 M1 的全局尺寸为 (A_shape[0], B_shape[1])
        engine2 = H5BlockMultiplicationEngine(
            self.temp_path + "M2.h5",
            swap,
            block_shape_A=(self.block_num, self.block_num),
            block_shape_B=(self.block_num, self.block_num),
            full_shape_A=(total_size, total_size),
            full_shape_B=(total_size, total_size),
            output_hdf5_filename=self.temp_path + "M3.h5",
            max_cache_blocks=self.max_cache_blocks,
            progress_desc="计算 M3 = M2@swap",
        )
        engine2.multiply()

        subtraction_engine = H5BlockSubtractionEngine(
            input_hdf5_filename=self.temp_path + "M3.h5",
            output_hdf5_filename=self.temp_path + "M4.h5",
            full_shape=(total_size, total_size),
            block_shape=(self.block_num, self.block_num),
            identity_scale=1.0,
            group_name="C",
            max_cache_blocks=self.max_cache_blocks,
            progress_desc="计算 M4 = M3 - I",
        )
        subtraction_engine.subtract_scaled_identity()
        subtraction_engine = H5BlockSubtractionEngine(
            input_hdf5_filename=self.temp_path + "M1.h5",
            output_hdf5_filename=self.temp_path + "M5.h5",
            full_shape=(total_size, total_size),
            block_shape=(self.block_num, self.block_num),
            identity_scale=1.0,
            group_name="C",
            max_cache_blocks=self.max_cache_blocks,
            progress_desc="计算 M5 = M1 - I",
        )
        subtraction_engine.subtract_scaled_identity()

        # R_4 = self.expm_A(1, "M4", total_size)
        # R_5 = self.expm_A(1, "M5", total_size)

        # engine_U = H5BlockMultiplicationEngine(
        #     R_4,
        #     R_5,
        #     block_shape_A=(self.block_num, self.block_num),
        #     block_shape_B=(self.block_num, self.block_num),
        #     full_shape_A=(total_size, total_size),
        #     full_shape_B=(total_size, total_size),
        #     output_hdf5_filename=u_output_hdf5_filename,
        #     max_cache_blocks=self.max_cache_blocks,
        #     progress_desc="计算 U = M4@M5",
        # )
        # engine_U.multiply()

        engine3 = H5BlockMultiplicationEngine(
            self.temp_path + "M4.h5",
            self.temp_path + "M5.h5",
            block_shape_A=(self.block_num, self.block_num),
            block_shape_B=(self.block_num, self.block_num),
            full_shape_A=(total_size, total_size),
            full_shape_B=(total_size, total_size),
            output_hdf5_filename=u_output_hdf5_filename,
            max_cache_blocks=self.max_cache_blocks,
            progress_desc="计算 U = M4@M5",
        )
        engine3.multiply()
        self.remove(self.temp_path + "M1.h5")
        self.remove(self.temp_path + "M2.h5")
        self.remove(self.temp_path + "M3.h5")
        self.remove(self.temp_path + "M4.h5")
        self.remove(self.temp_path + "M5.h5")
        # self.remove(R_5)
        # self.remove(R_4)

    def expm_A(self, theta, A_name, total_size):
        theta = np.pi * theta
        cosθ2 = math.cos(theta / 2)
        sinθ2 = math.sin(theta / 2)

        engine_a = H5BlockScalarMultiplicationEngine(
            self.temp_path + A_name + ".h5",
            self.temp_path + A_name + "_T1.h5",
            sinθ2,
            full_shape=(total_size, total_size),
            block_shape=(self.block_num, self.block_num),
            group_name="C",
            max_cache_blocks=self.max_cache_blocks,
            progress_desc="T1=sinθ2 * M5.h5",
        )
        engine_a.multiply_scalar()

        engine_b = H5BlockComplexMultiplyEngine(
            self.temp_path + A_name + "_T1.h5",
            self.temp_path + A_name + "T1_complex.h5",
            0,
            -1,
            full_shape=(total_size, total_size),
            block_shape=(self.block_num, self.block_num),
            group_name="C",
            max_cache_blocks=self.max_cache_blocks,
            progress_desc="T1_complex = −i·T1",
        )
        engine_b.multiply_complex()

        engine_c = H5BlockSubtractionEngine(
            input_hdf5_filename=self.temp_path + A_name + "T1_complex.h5",
            output_hdf5_filename=self.temp_path + A_name + "R5_theta.h5",
            full_shape=(total_size, total_size),
            block_shape=(self.block_num, self.block_num),
            identity_scale=-cosθ2,
            group_name="C",
            max_cache_blocks=self.max_cache_blocks,
            progress_desc="计算 R5_theta = T1_complex + cosθ2 * I",
        )
        engine_c.subtract_scaled_identity()
        self.remove(self.temp_path + A_name + "_T1.h5")
        self.remove(self.temp_path + A_name + "T1_complex.h5")
        return self.temp_path + A_name + "R5_theta.h5"

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

    def creatU(
        self,
        proj_output_hdf5_filename,
        swap_output_hdf5_filename,
        u_output_hdf5_filename,
    ):
        # 构造投影算符和交换算符
        self.create_proj_matrix_vectorized(proj_output_hdf5_filename)
        self.create_swap_matrix(swap_output_hdf5_filename)
        self.creat_u_operator_op(
            proj_output_hdf5_filename, swap_output_hdf5_filename, u_output_hdf5_filename
        )
        self.remove(proj_output_hdf5_filename)
        self.remove(swap_output_hdf5_filename)

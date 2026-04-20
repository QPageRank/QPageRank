import numpy as np
import os
import h5py
import logging
from tqdm import tqdm
from scipy.sparse import eye

from scipy.sparse import (
    csr_matrix,
    random as sparse_random,
    eye,
    lil_matrix,
    dia_matrix,
)


#############################
# 辅助函数：缓存写入 HDF5
#############################
def flush_cache_sparse(cache, h5group):
    """
    将缓存中保存的稀疏矩阵块（csr_matrix）写入 HDF5 文件中的 h5group。
    每个块在 h5group 下创建子组，名称格式为 "block_{row_start}_{col_start}"，
    存储 data、indices、indptr 以及块的 shape、row_start 和 col_start 属性。
    对于每个块，若组已经存在，则先删除再创建。
    """
    for (row_start, col_start, row_end, col_end), block in cache.items():
        group_name = f"block_{row_start}_{col_start}"
        if group_name in h5group:
            del h5group[group_name]
        block_group = h5group.create_group(group_name)
        block_group.create_dataset("data", data=block.data)
        block_group.create_dataset("indices", data=block.indices)
        block_group.create_dataset("indptr", data=block.indptr)
        block_group.attrs["shape"] = (row_end - row_start, col_end - col_start)
        block_group.attrs["row_start"] = row_start
        block_group.attrs["col_start"] = col_start
    cache.clear()


#############################
# 辅助函数：加载和合并结果块
#############################
def load_result_blocks(filename, group_name="C"):
    blocks = []
    with h5py.File(filename, "r") as f:
        C_group = f[group_name]
        for block_key in C_group.keys():
            block_group = C_group[block_key]
            data = block_group["data"][:]
            indices = block_group["indices"][:]
            indptr = block_group["indptr"][:]
            shape = tuple(block_group.attrs["shape"])
            row_start = block_group.attrs["row_start"]
            col_start = block_group.attrs["col_start"]
            csr_block = csr_matrix((data, indices, indptr), shape=shape)
            blocks.append((row_start, col_start, csr_block))
    return blocks


def combine_result_blocks(result_blocks, full_shape):
    full_matrix = np.zeros(full_shape, dtype=np.float32)
    for row_start, col_start, block in result_blocks:
        block_dense = block.toarray()
        r, c = block_dense.shape
        full_matrix[row_start : row_start + r, col_start : col_start + c] = block_dense
    return full_matrix


#############################
# 内存中稀疏矩阵分块类（懒加载方式）
#############################
class BlockSparseMatrix:
    def __init__(self, matrix, block_shape):
        self.matrix = matrix
        self.block_shape = block_shape
        self.shape = matrix.shape
        self.num_row_blocks = (self.shape[0] + block_shape[0] - 1) // block_shape[0]
        self.num_col_blocks = (self.shape[1] + block_shape[1] - 1) // block_shape[1]

    def get_block(self, i, j):
        block_rows, block_cols = self.block_shape
        row_start = i * block_rows
        row_end = min((i + 1) * block_rows, self.shape[0])
        col_start = j * block_cols
        col_end = min((j + 1) * block_cols, self.shape[1])
        # 强制转换为 CSR 格式，确保内部结构正确
        return self.matrix[row_start:row_end, col_start:col_end].tocsr()

    @classmethod
    def from_matrix(cls, matrix, block_shape):
        return cls(matrix, block_shape)

    def write_to_h5(self, h5_filename, group_name="C", max_cache_blocks=20):
        if os.path.dirname(h5_filename) != "":
            os.makedirs(os.path.dirname(h5_filename), exist_ok=True)
        with h5py.File(h5_filename, "w") as f:
            grp = f.create_group(group_name)
            cache = {}
            for i in range(self.num_row_blocks):
                for j in range(self.num_col_blocks):
                    block = self.get_block(i, j)
                    row_start = i * self.block_shape[0]
                    col_start = j * self.block_shape[1]
                    actual_shape = block.shape
                    cache[
                        (
                            row_start,
                            col_start,
                            row_start + actual_shape[0],
                            col_start + actual_shape[1],
                        )
                    ] = block
                    if len(cache) >= max_cache_blocks:
                        flush_cache_sparse(cache, grp)
            if cache:
                flush_cache_sparse(cache, grp)
        print(f"分块矩阵已写入文件：{h5_filename}")


#############################
# HDF5 分块矩阵类（用于加载文件中的分块）
#############################
class H5BlockMatrix:
    def __init__(self, h5_filename, group_name="C", block_shape=None, full_shape=None):
        self.h5_filename = h5_filename
        self.group_name = group_name
        self.block_shape = block_shape
        self.full_shape = full_shape

    def get_block(self, i, j):
        row_start = i * self.block_shape[0]
        col_start = j * self.block_shape[1]
        key = f"block_{row_start}_{col_start}"
        with h5py.File(self.h5_filename, "r") as f:
            group = f[self.group_name]
            if key in group:
                block_group = group[key]
                data = block_group["data"][:]
                indices = block_group["indices"][:]
                indptr = block_group["indptr"][:]
                shape = tuple(block_group.attrs["shape"])
                return csr_matrix((data, indices, indptr), shape=shape)
            else:
                return None

    def num_blocks(self):
        num_row_blocks = (
            self.full_shape[0] + self.block_shape[0] - 1
        ) // self.block_shape[0]
        num_col_blocks = (
            self.full_shape[1] + self.block_shape[1] - 1
        ) // self.block_shape[1]
        return num_row_blocks, num_col_blocks


#############################
# H5BlockMultiplicationEngine 类（支持 A 与 B 分块尺寸不同）
#############################
class H5BlockMultiplicationEngine:
    def __init__(
        self,
        h5_filename_A,
        h5_filename_B,
        block_shape_A,
        block_shape_B,
        full_shape_A,
        full_shape_B,
        output_hdf5_filename,
        max_cache_blocks=10,
        progress_desc="HDF5 分块乘法",
    ):
        if block_shape_A[1] != block_shape_B[0]:
            raise ValueError("A 的分块宽度必须等于 B 的分块高度！")
        self.h5_filename_A = h5_filename_A
        self.h5_filename_B = h5_filename_B
        self.block_shape_A = block_shape_A
        self.block_shape_B = block_shape_B
        self.full_shape_A = full_shape_A
        self.full_shape_B = full_shape_B
        self.output_hdf5_filename = output_hdf5_filename
        self.max_cache_blocks = max_cache_blocks
        self.progress_desc = progress_desc
        self.logger = logging.getLogger(self.__class__.__name__)

        self.A = H5BlockMatrix(
            h5_filename_A,
            group_name="C",
            block_shape=block_shape_A,
            full_shape=full_shape_A,
        )
        self.B = H5BlockMatrix(
            h5_filename_B,
            group_name="C",
            block_shape=block_shape_B,
            full_shape=full_shape_B,
        )

    def multiply(self):
        num_row_blocks_A = (
            self.full_shape_A[0] + self.block_shape_A[0] - 1
        ) // self.block_shape_A[0]
        num_col_blocks_A = (
            self.full_shape_A[1] + self.block_shape_A[1] - 1
        ) // self.block_shape_A[1]
        num_row_blocks_B = (
            self.full_shape_B[0] + self.block_shape_B[0] - 1
        ) // self.block_shape_B[0]
        num_col_blocks_B = (
            self.full_shape_B[1] + self.block_shape_B[1] - 1
        ) // self.block_shape_B[1]

        if num_col_blocks_A != num_row_blocks_B:
            raise ValueError("A 的列块数必须等于 B 的行块数！")

        total = (num_row_blocks_A) * (num_col_blocks_B)
        pbar = tqdm(total=total, desc=self.progress_desc)
        cache = {}
        self.logger.info(
            f"Starting multiply(): in_path={self.h5_filename_A}, out_path={self.h5_filename_B}"
        )
        for i in range(num_row_blocks_A):

            for j in range(num_col_blocks_B):
                self.logger.debug(f"Loading block ({i} ,{j})")
                A_block_ex = self.A.get_block(i, 0)
                B_block_ex = self.B.get_block(0, j)
                if A_block_ex is None or B_block_ex is None:
                    continue
                block_rows = A_block_ex.shape[0]
                block_cols = B_block_ex.shape[1]
                C_block = csr_matrix((block_rows, block_cols), dtype=np.float32)
                num_inner = num_col_blocks_A
                for k in range(num_inner):
                    A_block = self.A.get_block(i, k)
                    B_block = self.B.get_block(k, j)
                    if A_block is None or B_block is None:
                        continue
                    C_block = C_block + A_block.dot(B_block)
                row_start = i * self.block_shape_A[0]
                col_start = j * self.block_shape_B[1]
                row_end = row_start + block_rows
                col_end = col_start + block_cols
                cache[(row_start, col_start, row_end, col_end)] = C_block
                if len(cache) >= self.max_cache_blocks:
                    self.logger.info(f"flushing to '{self.output_hdf5_filename}'")
                    self._flush_cache(cache)
                    self.logger.debug("Flush successful, clearing cache")
                pbar.update(1)
        if cache:
            self._flush_cache(cache)
            self.logger.debug("Final flush successful, clearing cache")
        pbar.close()
        print("HDF5 分块乘法完成，结果存储在文件：", self.output_hdf5_filename)

    def _flush_cache(self, cache):
        folder = os.path.dirname(self.output_hdf5_filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        with h5py.File(self.output_hdf5_filename, "a") as f:
            if "C" not in f:
                C_group = f.create_group("C")
            else:
                C_group = f["C"]
            flush_cache_sparse(cache, C_group)

    def load_result_blocks(self, group_name="C"):
        return load_result_blocks(self.output_hdf5_filename, group_name=group_name)

    def get_combined_result(self):
        result_blocks = self.load_result_blocks()
        full_shape = (self.full_shape_A[0], self.full_shape_B[1])
        return combine_result_blocks(result_blocks, full_shape)

    def get_block_results(self):
        return self.load_result_blocks()


def get_start_state(start_path, shape_A_0, shape_B_1, group_name="C"):
    result_blocks = load_result_blocks(start_path, group_name=group_name)
    full_shape = (shape_A_0, shape_B_1)
    return combine_result_blocks(result_blocks, full_shape)


#############################
# H5BlockScalarMultiplicationEngine 类（常数乘法）
#############################
class H5BlockScalarMultiplicationEngine:
    def __init__(
        self,
        input_hdf5_filename,
        output_hdf5_filename,
        scalar,
        full_shape,
        block_shape,
        group_name="C",
        max_cache_blocks=10,
        progress_desc="HDF5 分块常数乘法",
    ):
        self.input_hdf5_filename = input_hdf5_filename
        self.output_hdf5_filename = output_hdf5_filename
        self.scalar = scalar
        self.full_shape = full_shape
        self.block_shape = block_shape
        self.group_name = group_name
        self.max_cache_blocks = max_cache_blocks
        self.progress_desc = progress_desc

    def multiply_scalar(self):
        m_blocks = load_result_blocks(
            self.input_hdf5_filename, group_name=self.group_name
        )
        total = len(m_blocks)
        pbar = tqdm(total=total, desc=self.progress_desc)
        cache = {}
        folder = os.path.dirname(self.output_hdf5_filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        with h5py.File(self.output_hdf5_filename, "w") as f:
            out_group = f.create_group(self.group_name)
            for row_start, col_start, block in m_blocks:
                new_block = self.scalar * block
                r, c = new_block.shape
                cache[(row_start, col_start, row_start + r, col_start + c)] = new_block
                if len(cache) >= self.max_cache_blocks:
                    flush_cache_sparse(cache, out_group)
                pbar.update(1)
            if cache:
                flush_cache_sparse(cache, out_group)
            pbar.close()
        print("HDF5 分块常数乘法完成，结果存储在文件：", self.output_hdf5_filename)

    def load_result_blocks(self):
        return load_result_blocks(self.output_hdf5_filename, group_name=self.group_name)

    def get_combined_result(self):
        blocks = self.load_result_blocks()
        return combine_result_blocks(blocks, self.full_shape)


class H5BlockComplexMultiplyEngine:
    def __init__(
        self,
        input_hdf5_filename: str,
        output_hdf5_filename: str,
        scalar_real: float,
        scalar_imag: float,
        full_shape: tuple,
        block_shape: tuple,
        group_name: str = "C",
        max_cache_blocks: int = 10,
        progress_desc: str = "HDF5 Block Complex Multiplication",
    ):
        self.input_hdf5_filename = input_hdf5_filename
        self.output_hdf5_filename = output_hdf5_filename
        # Combine real and imaginary parts into one complex scalar
        self.scalar = scalar_real + 1j * scalar_imag
        self.full_shape = full_shape
        self.block_shape = block_shape
        self.group_name = group_name
        self.max_cache_blocks = max_cache_blocks
        self.progress_desc = progress_desc

    def multiply_complex(self):
        # 1) 读取已有块
        m_blocks = load_result_blocks(
            self.input_hdf5_filename, group_name=self.group_name
        )
        total = len(m_blocks)
        pbar = tqdm(total=total, desc=self.progress_desc)
        cache = {}

        # 2) 打开输出 HDF5 文件并准备组
        with h5py.File(self.output_hdf5_filename, "w") as f_out:
            out_group = f_out.create_group(self.group_name)

            # 3) 对每个块乘以复数标量并缓存写入
            for row_start, col_start, block in m_blocks:
                # 转为 complex128 后乘以复数标量
                complex_block = block.astype(np.complex128) * self.scalar
                r, c = complex_block.shape
                cache[(row_start, col_start, row_start + r, col_start + c)] = (
                    complex_block
                )

                if len(cache) >= self.max_cache_blocks:
                    flush_cache_sparse(cache, out_group)
                    cache.clear()
                pbar.update(1)

            # 4) 刷新剩余缓存
            if cache:
                flush_cache_sparse(cache, out_group)
            pbar.close()

        print(
            "HDF5 Block Complex Multiplication complete. Result at:",
            self.output_hdf5_filename,
        )

    def load_result_blocks(self):
        return load_result_blocks(self.output_hdf5_filename, group_name=self.group_name)

    def get_combined_result(self):
        blocks = self.load_result_blocks()
        return combine_result_blocks(blocks, self.full_shape)


#############################
# H5BlockSubtractionEngine 类（分块加减法，可调单位矩阵倍数）
#############################
class H5BlockSubtractionEngine:
    def __init__(
        self,
        input_hdf5_filename: str,
        output_hdf5_filename: str,
        full_shape: tuple,
        block_shape: tuple,
        identity_scale: float = 1.0,
        group_name: str = "C",
        max_cache_blocks: int = 10,
        progress_desc: str = "HDF5 分块加减法",
    ):
        """
        :param identity_scale: 要减去的单位矩阵倍数 α（计算 M - α·I）
        """
        self.input_hdf5_filename = input_hdf5_filename
        self.output_hdf5_filename = output_hdf5_filename
        self.full_shape = full_shape
        self.block_shape = block_shape
        self.identity_scale = identity_scale
        self.group_name = group_name
        self.max_cache_blocks = max_cache_blocks
        self.progress_desc = progress_desc

    def subtract_scaled_identity(self):
        """
        对输入 HDF5 中的分块矩阵 M 计算 X = M - α·I，
        其中 I 为与 M 同尺寸的单位矩阵，α = self.identity_scale。
        仅对主对角块执行减法，其它块原样写出。
        """
        m_blocks = load_result_blocks(
            self.input_hdf5_filename, group_name=self.group_name
        )
        total = len(m_blocks)
        pbar = tqdm(total=total, desc=self.progress_desc)
        cache = {}
        with h5py.File(self.output_hdf5_filename, "w") as f_out:
            X_group = f_out.create_group(self.group_name)
            for row_start, col_start, M_block in m_blocks:
                r, c = M_block.shape
                if row_start == col_start:
                    # 构造 α·I
                    I_block = (
                        eye(r, format="csr", dtype=M_block.dtype) * self.identity_scale
                    )
                    X_block = M_block - I_block
                else:
                    X_block = M_block
                cache[(row_start, col_start, row_start + r, col_start + c)] = X_block
                if len(cache) >= self.max_cache_blocks:
                    flush_cache_sparse(cache, X_group)
                    cache.clear()
                pbar.update(1)
            if cache:
                flush_cache_sparse(cache, X_group)
            pbar.close()
        print(
            f"HDF5 分块加减法完成 (M - {self.identity_scale}·I)，结果存储在：{self.output_hdf5_filename}"
        )

    def load_result_blocks(self):
        return load_result_blocks(self.output_hdf5_filename, group_name=self.group_name)

    def get_combined_result(self):
        blocks = self.load_result_blocks()
        return combine_result_blocks(blocks, self.full_shape)


#############################
# SparseMultiplicationEngine 类（整体流程封装）
#############################
class SparseMultiplicationEngine:
    def __init__(
        self,
        A,
        B,
        block_shape_A,
        block_shape_B,
        hdf5_filename,
        max_cache_blocks=10,
        progress_desc="稀疏矩阵分块乘法",
    ):
        self.A = A
        self.B = B
        self.block_shape_A = block_shape_A
        self.block_shape_B = block_shape_B
        self.hdf5_filename = hdf5_filename
        self.max_cache_blocks = max_cache_blocks
        self.progress_desc = progress_desc

    def run(self):
        # 将 A 分块写入 HDF5
        A_block = BlockSparseMatrix.from_matrix(self.A, self.block_shape_A)
        A_block.write_to_h5(
            "Data/Temp/A_blocks.h5",
            group_name="C",
            max_cache_blocks=self.max_cache_blocks,
        )
        # 将 B 分块写入 HDF5
        B_block = BlockSparseMatrix.from_matrix(self.B, self.block_shape_B)
        B_block.write_to_h5(
            "Data/Temp/B_blocks.h5",
            group_name="C",
            max_cache_blocks=self.max_cache_blocks,
        )
        # 使用 H5BlockMultiplicationEngine 进行乘法
        engine = H5BlockMultiplicationEngine(
            "Data/Temp/A_blocks.h5",
            "Data/Temp/B_blocks.h5",
            block_shape_A=self.block_shape_A,
            block_shape_B=self.block_shape_B,
            full_shape_A=self.A.shape,
            full_shape_B=self.B.shape,
            output_hdf5_filename=self.hdf5_filename,
            max_cache_blocks=self.max_cache_blocks,
            progress_desc=self.progress_desc,
        )
        engine.multiply()
        self.remove("Data/Temp/A_blocks.h5")
        self.remove("Data/Temp/B_blocks.h5")
        print("计算完成，结果存储在文件：", self.hdf5_filename)

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

    def load_result_blocks(self):
        return load_result_blocks(self.hdf5_filename, group_name="C")

    def get_combined_result(self):
        result_blocks = self.load_result_blocks()
        full_shape = (self.A.shape[0], self.B.shape[1])
        return combine_result_blocks(result_blocks, full_shape)

    def get_block_results(self):
        return self.load_result_blocks()

    def multiply_result_blocks_with_D(
        self, D, output_hdf5_filename, max_cache_blocks=10, progress_desc="后续分块乘法"
    ):
        block_results = self.get_block_results()
        total = len(block_results)
        pbar = tqdm(total=total, desc=progress_desc)
        new_cache = {}
        with h5py.File(output_hdf5_filename, "w") as f:
            Y_group = f.create_group("Y")
            for row_start, col_start, X_block in block_results:
                Y_block = X_block.dot(D)
                r, c = Y_block.shape
                new_cache[(row_start, col_start, row_start + r, col_start + c)] = (
                    Y_block
                )
                if len(new_cache) >= max_cache_blocks:
                    flush_cache_sparse(new_cache, Y_group)
                pbar.update(1)
            if new_cache:
                flush_cache_sparse(new_cache, Y_group)
            pbar.close()
        print("后续乘法计算完成，结果存储在文件：", output_hdf5_filename)
        return output_hdf5_filename

    def get_combined_result_from_file(self, filename, group_name="Y"):
        result_blocks = load_result_blocks(filename, group_name=group_name)
        full_shape = (self.A.shape[0], self.B.shape[1])  # 根据 D 的形状可能需要调整
        return combine_result_blocks(result_blocks, full_shape)


# #############################
# # 主程序：测试代码正确性
# #############################
# if __name__ == "__main__":
#     # 调整矩阵尺寸为 1000x1000，块尺寸为 250x250
#     A_shape = (1000, 1000)
#     B_shape = (1000, 1000)
#     density_A = 1e-2
#     density_B = 1e-2

#     A_sparse = sparse_random(A_shape[0], A_shape[1], density=density_A, format="csr",
#                              dtype=np.float32, random_state=42)
#     B_sparse = sparse_random(B_shape[0], B_shape[1], density=density_B, format="csr",
#                              dtype=np.float32, random_state=24)

#     output_hdf5_filename = "C_product.h5"

#     engine = SparseMultiplicationEngine(A_sparse, B_sparse,
#                                           block_shape_A=(250, 250),
#                                           block_shape_B=(250, 250),
#                                           hdf5_filename=output_hdf5_filename,
#                                           max_cache_blocks=10,
#                                           progress_desc="HDF5 分块乘法测试")
#     engine.run()

#     C_combined = engine.get_combined_result()
#     C_direct = A_sparse.dot(B_sparse).toarray()
#     if np.allclose(C_direct, C_combined, atol=1e-5):
#         print("验证通过：HDF5 分块乘法结果与直接乘法结果一致！")
#     else:
#         print("验证失败：结果不一致。")

#     block_results = engine.get_block_results()
#     print("加载分块结果：")
#     for row_start, col_start, block in block_results:
#         print(f"块起始位置: ({row_start}, {col_start}), 形状: {block.shape}, 非零数: {block.nnz}")

#     # 示例：将分块结果与另一个稀疏矩阵 D 相乘
#     D_sparse = sparse_random(B_shape[1], 500, density=1e-2, format="csr",
#                              dtype=np.float32, random_state=99)
#     output_D_h5 = "Y_product.h5"
#     engine.multiply_result_blocks_with_D(D_sparse, output_D_h5, max_cache_blocks=10,
#                                           progress_desc="后续分块乘法测试")
#     Y_combined = engine.get_combined_result_from_file(output_D_h5, group_name="Y")
#     Y_direct = C_direct.dot(D_sparse.toarray())
#     if np.allclose(Y_direct, Y_combined, atol=1e-5):
#         print("验证通过：后续乘法结果与直接乘法结果一致！")
#     else:
#         print("验证失败：后续乘法结果不一致。")

#     # 示例：利用 H5BlockSubtractionEngine 对 C_product.h5 计算 X = 2*M - I
#     subtraction_engine = H5BlockSubtractionEngine(
#         input_hdf5_filename=output_hdf5_filename,
#         output_hdf5_filename="X_product.h5",
#         full_shape=A_shape,  # 假设 M 为方阵，与 A 的尺寸一致
#         block_shape=(250, 250),
#         group_name="C",
#         max_cache_blocks=10,
#         progress_desc="计算 X = 2*M - I"
#     )
#     subtraction_engine.subtract_identity()
#     X_combined = subtraction_engine.get_combined_result()
#     M_direct = A_sparse.dot(B_sparse).toarray()
#     I = eye(A_shape[0], format="csr", dtype=np.float32).toarray()
#     X_direct = 2 * M_direct - I
#     if np.allclose(X_direct, X_combined, atol=1e-5):
#         print("验证通过：X 计算结果一致！")
#     else:
#         print("验证失败：X 计算结果不一致！")

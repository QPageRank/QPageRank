from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
import numpy as np
from qiskit.circuit.library import MCXGate, RYGate


def _calc_rotation_angles(prob_dist):
    """根据概率分布递归二分计算旋转角度（辅助工具函数）"""
    angles = []

    def _recurse(probs, depth=0):
        if len(probs) == 1:
            return
        p0 = sum(probs[: len(probs) // 2])
        p1 = sum(probs[len(probs) // 2 :])
        angle = 2 * np.arcsin(np.sqrt(p0 / (p0 + p1))) if (p0 + p1) != 0 else 0
        if depth >= len(angles):
            angles.append(angle)
        else:
            angles[depth] = angle
        _recurse(probs[: len(probs) // 2], depth + 1)
        _recurse(probs[len(probs) // 2 :], depth + 1)

    _recurse(prob_dist)
    return angles


def _add_multi_controlled_hadamard(qc, q_aux, q_current, target_reg, ctrl_str):
    """
    添加多控哈达玛门：
    对目标寄存器中每个比特施加受控 RY 旋转门，以近似实现 Hadamard 门操作。
    参数：
        qc : QuantumCircuit - 当前电路
        q_aux : QuantumRegister - 辅助量子位（在控制条件中使用）
        q_current : QuantumRegister - 当前节点寄存器（参与控制）
        target_reg : QuantumRegister - 目标寄存器（受门操作作用）
        ctrl_str : str - 控制条件字符串（长度应为 len(q_current)+1）
    """
    n = len(target_reg)
    hadamard_angle = np.pi / 2  # Hadamard 操作对应的旋转角度
    for i in range(n):
        mcry_gate = RYGate(hadamard_angle).control(
            len(q_current) + 1, ctrl_state=ctrl_str
        )
        qc.append(mcry_gate, list(q_current) + [q_aux[0]] + [target_reg[i]], [])


def _add_multi_controlled_rotation(qc, angles, q_aux, q_current, target_reg, ctrl_str):
    """
    添加多控旋转门：
    对目标寄存器的每个比特施加受控 RY 门，旋转角度由 angles 给出。
    参数：
        qc : QuantumCircuit - 当前电路
        angles : list - 与目标寄存器比特数相同的旋转角度列表
        q_aux : QuantumRegister - 辅助量子位（在控制条件中使用）
        q_current : QuantumRegister - 当前节点寄存器（参与控制）
        target_reg : QuantumRegister - 目标寄存器（施加旋转门）
        ctrl_str : str - 控制条件字符串（长度应为 len(q_current)+1）
    """
    n = len(target_reg)
    if len(angles) != n:
        raise ValueError("角度列表的长度必须与目标寄存器的量子比特数一致")
    for i in range(n):
        angle = angles[i]
        mcry_gate = RYGate(angle).control(len(q_current) + 1, ctrl_state=ctrl_str)
        qc.append(mcry_gate, list(q_current) + [q_aux[0]] + [target_reg[i]], [])


def build_initial_state_circuit(G, alpha):
    """
    构造初始态制备模块（对应量子PageRank的初态构造）
    参数：
        G : np.array - 转移矩阵（假设为列随机矩阵），形状为 (N,N)
        alpha : float - 阻尼系数（例如 0.85）
    返回：
        qc : QuantumCircuit - 包含辅助位、当前节点寄存器和下一节点寄存器的初态制备电路
        q_aux, q_current, q_next : 对应的量子寄存器
    """
    N = G.shape[0]  # 节点数
    n = int(np.ceil(np.log2(N)))  # 为表示 N 个节点需要的比特数
    q_aux = QuantumRegister(1, "aux")  # 辅助位 (用于编码阻尼信息)
    q_current = QuantumRegister(n, "q_curr")  # 当前节点寄存器
    q_next = QuantumRegister(n, "q_next")  # 下一节点寄存器
    qc = QuantumCircuit(q_aux, q_current, q_next)

    # 1. 辅助位初始化：使用 RY 门使得辅助位状态为 sqrt(alpha)|0> + sqrt(1-alpha)|1>
    theta_alpha_val = 2 * np.arccos(
        np.sqrt(alpha)
    )  # 满足 cos(theta_alpha/2) = sqrt(alpha)
    qc.ry(theta_alpha_val, q_aux)

    # theta_alpha_val = 2 * np.arcsin(np.sqrt(1 - alpha))  # 正确公式
    # qc.ry(theta_alpha_val, q_aux)

    # 2. 当前节点寄存器均匀叠加（构造均匀叠加态）
    qc.h(q_current)

    # 3. 条件转移态制备：
    # 对于每个节点 j，控制条件为辅助位和当前节点的状态
    # 此处我们构造两种控制操作：
    #   - 当辅助位为 0 时，根据转移矩阵 G 的第 j 列加载特定概率幅（使用多控旋转门）
    #   - 当辅助位为 1 时，加载均匀分布（使用多控 Hadamard 近似操作）
    for j in range(N):
        bin_j = format(j, f"0{n}b")  # j 的 n 位二进制表示
        ctrl_str_0 = "0" + bin_j  # 控制条件：aux = 0 且 q_current = bin_j
        ctrl_str_1 = "1" + bin_j  # 控制条件：aux = 1 且 q_current = bin_j
        # 设 G[:, j] 的值可以直接用作旋转角度（实际中可能需要转换）
        angles = _calc_rotation_angles(G[:, j])
        _add_multi_controlled_rotation(qc, angles, q_aux, q_current, q_next, ctrl_str_0)
        _add_multi_controlled_hadamard(qc, q_aux, q_current, q_next, ctrl_str_1)

    qc.barrier()
    return qc, q_aux, q_current, q_next


def build_reflection_operator(Upsi, n):
    """
    构造反射算符 R = Upsi * (2|0⟩⟨0| - I) * Upsi† 的子电路。
    参数：
        Upsi : QuantumCircuit - 子电路（作用在目标寄存器上），满足 Upsi|0...0⟩ = |\psi⟩
        n : int - 目标寄存器的比特数
    返回：
        QuantumCircuit - 反射操作子电路，作用在一个 n 比特寄存器上
    """
    qc_reflect = QuantumCircuit(Upsi.num_qubits, name="Reflect")
    # 模块 A: 应用 Upsi† (逆制备)
    qc_reflect.append(Upsi.inverse(), qc_reflect.qubits)
    qc_reflect.barrier()
    # 模块 B: 在标准基上施加反射 R0 = 2|0...0⟩⟨0...0| - I
    qc_reflect.h(range(n))
    # 使用多控 X 门实现条件相移：当所有控制比特为 |0⟩ 时施加 Z 相移
    mcx_gate = MCXGate(num_ctrl_qubits=n - 1, ctrl_state="0" * (n - 1))
    qc_reflect.append(mcx_gate, qc_reflect.qubits[:n])  # 确保只作用于前 n 个比特
    qc_reflect.h(range(n))
    qc_reflect.barrier()
    # 模块 C: 恢复 Upsi
    qc_reflect.append(Upsi, qc_reflect.qubits)
    qc_reflect.barrier()
    return qc_reflect


def build_shift_operator(q_current, q_next):
    """
    构造移位算符，将当前节点寄存器与下一节点寄存器逐比特交换。
    参数：
        q_current : QuantumRegister - 当前节点寄存器
        q_next : QuantumRegister - 下一节点寄存器
    返回：
        QuantumCircuit - 实现 SWAP 操作的移位模块
    """
    qc_shift = QuantumCircuit(q_current, q_next, name="Shift")
    n = len(q_current)
    for i in range(n):
        qc_shift.swap(q_current[i], q_next[i])
    qc_shift.barrier()
    return qc_shift


def build_qpagerank_circuit(G, alpha=0.85, steps=2):
    """
    构建量子PageRank算法电路。步骤包括：
      1. 利用转移矩阵 G 和阻尼参数 alpha 构造初始态制备模块。
      2. 构造反射算符模块（基于子电路 Upsi）。
      3. 构造移位算符模块（SWAP操作）。
      4. 重复执行演化操作 U = Shift * Reflect steps 次。
      5. 测量当前节点寄存器得到排序结果。
    参数：
        G : np.array - 转移矩阵 (列随机矩阵)，形状为 (N, N)
        alpha : float - 阻尼系数（例如 0.85）
        steps : int - 演化步骤数
    返回：
        QuantumCircuit - 完整的量子PageRank电路
    """
    # 1. 初态制备模块
    init_circ, q_aux, q_curr, q_next = build_initial_state_circuit(G, alpha)
    n = len(q_curr)
    cr = ClassicalRegister(n, "c")
    qc = QuantumCircuit(q_aux, q_curr, q_next, cr)

    # 添加初态制备子电路
    qc.append(init_circ.to_instruction(), q_aux[:] + q_curr[:] + q_next[:])
    qc.barrier()

    # 2. 构造 Upsi 子电路用于反射算符：
    # 这里我们提取一个简单的子电路：对当前节点寄存器施加 Hadamard 门（构造均匀叠加态）
    Upsi = QuantumCircuit(q_curr, q_next, name="Upsi")
    Upsi.h(q_curr)  # 简单起见，仅对 q_curr 施加 Hadamard
    Upsi.barrier()

    # 3. 构造反射算符模块（作用在 q_curr 和 q_next 的组合寄存器上）
    # 为简化，我们只对 q_curr 寄存器进行反射操作
    reflect_circ = build_reflection_operator(Upsi, n)

    # 4. 构造移位算符（交换 q_curr 与 q_next 寄存器）
    shift_circ = build_shift_operator(q_curr, q_next)

    # 5. 演化操作：重复执行 steps 次，执行 U = Shift * Reflect
    for _ in range(steps):
        qc.append(reflect_circ.to_instruction(), q_curr[:] + q_next[:])
        qc.append(shift_circ.to_instruction(), q_curr[:] + q_next[:])
        qc.barrier()

    # 6. 测量当前节点寄存器，将结果存储到经典寄存器
    qc.measure(q_curr, cr)
    qc.barrier()
    return qc


# ================== 使用示例 ==================
if __name__ == "__main__":
    # 示例转移矩阵 (4节点)
    G = np.array(
        [
            [0.0375, 0.4625, 0.4625, 0.0375],
            [0.4625, 0.0375, 0.4625, 0.0375],
            [0.4625, 0.4625, 0.0375, 0.0375],
            [0.0375, 0.0375, 0.0375, 0.8875],
        ]
    )

    # 构建完整的量子PageRank电路
    qc = build_qpagerank_circuit(G, alpha=0.85, steps=2)

    # 绘制电路图并保存为图片
    fig = qc.decompose().draw(output="mpl", fold=-1, scale=0.8)
    fig.savefig("q_pagerank_circuit.png", bbox_inches="tight")

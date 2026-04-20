import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import MCPhaseGate, RYGate, ZGate
from qiskit.quantum_info import Statevector
import os
import sys

sys.path.append(os.getcwd())
from RoadSelect.Quantum.CreatProbabilityMatrix import CreatProbabilityMatrix

# -------------------------------
# 辅助函数（与前面版本基本相同）
# -------------------------------


def _calc_rotation_angles(prob_dist):
    """计算概率分布对应的旋转角度，递归二分法计算"""
    angles = []

    def _recurse(probs, depth=0):
        if len(probs) == 1:
            return
        p0 = sum(probs[: len(probs) // 2])
        p1 = sum(probs[len(probs) // 2 :])
        angle = 2 * np.arccos(np.sqrt(p0 / (p0 + p1))) if (p0 + p1) != 0 else 0
        if depth >= len(angles):
            angles.append(angle)
        else:
            angles[depth] = angle
        _recurse(probs[: len(probs) // 2], depth + 1)
        _recurse(probs[len(probs) // 2 :], depth + 1)

    _recurse(prob_dist)
    return angles


def _add_multi_controlled_hadamard(qc, q_aux, q_current, target_reg, ctrl_str):
    """添加多控Hadamard门（利用 RY(pi/2) 实现）"""
    n = len(target_reg)
    hadamard_angle = np.pi / 2  # RY(π/2) 实现 Hadamard
    if len(ctrl_str) != len(q_current) + 1:
        raise ValueError("控制字符串长度错误")
    for i in range(n):
        mcry_gate = RYGate(hadamard_angle).control(
            len(q_current) + 1, ctrl_state=ctrl_str
        )
        qc.append(mcry_gate, list(q_current) + [q_aux[0]] + [target_reg[i]], [])


def _add_multi_controlled_rotation(qc, angles, q_aux, q_current, target_reg, ctrl_str):
    """添加多控旋转门，将每个目标比特施加指定旋转角"""
    n = len(target_reg)
    if len(angles) != n:
        raise ValueError("角度列表长度错误")
    if len(ctrl_str) != len(q_current) + 1:
        raise ValueError("控制字符串长度错误")
    for i in range(n):
        angle = angles[i]
        mcry_gate = RYGate(angle).control(len(q_current) + 1, ctrl_state=ctrl_str)
        qc.append(mcry_gate, list(q_current) + [q_aux[0]] + [target_reg[i]], [])


# -------------------------------
# 核心电路构建
# -------------------------------


def build_initial_state_circuit(G, alpha):
    """
    构造初始态制备电路：
      - q_aux 记录阻尼系数信息；
      - q_current 表示当前节点（用二进制编码）；
      - q_next 表示下一节点。
    利用受控旋转/受控Hadamard门根据转移矩阵 G 将信息加载到 q_next 中。
    """
    N = G.shape[0]
    n = int(np.ceil(np.log2(N)))
    q_aux = QuantumRegister(1, "aux")
    q_current = QuantumRegister(n, "q_curr")
    q_next = QuantumRegister(n, "q_next")

    qc = QuantumCircuit(q_aux, q_current, q_next)

    # 辅助位初始化: RY 门，角度为 2 * arcsin(sqrt(1-alpha))
    theta_alpha_val = 2 * np.arcsin(np.sqrt(1 - alpha))
    qc.ry(theta_alpha_val, q_aux)
    qc.h(q_current)  # 当前节点均匀叠加

    for j in range(N):
        bin_j = format(j, f"0{n}b")
        ctrl_str_0 = "0" + bin_j  # 辅助位为0的情况
        ctrl_str_1 = "1" + bin_j  # 辅助位为1的情况
        angles = _calc_rotation_angles(G[:, j])
        _add_multi_controlled_rotation(qc, angles, q_aux, q_current, q_next, ctrl_str_0)
        _add_multi_controlled_hadamard(qc, q_aux, q_current, q_next, ctrl_str_1)
    qc.barrier()
    return qc, [q_aux[0]] + list(q_current) + list(q_next)


def build_reflection_operator(init_circuit):
    total_qubits = init_circuit.qubits
    reflect_qreg = QuantumRegister(len(total_qubits), "reflect_qreg")
    qc_reflect = QuantumCircuit(reflect_qreg, name="Reflect")

    # 逆初始化
    qc_reflect.append(init_circuit.inverse(), reflect_qreg)

    # Grover扩散（针对全零态的反射）
    qc_reflect.h(reflect_qreg)
    qc_reflect.x(reflect_qreg)  # 将|0⟩转换为|1⟩以匹配控制条件
    # 应用多控制Z门（控制全1时触发相位翻转）
    z_gate = ZGate().control(
        len(reflect_qreg) - 1, ctrl_state="1" * (len(reflect_qreg) - 1)
    )
    qc_reflect.append(z_gate, reflect_qreg)
    qc_reflect.x(reflect_qreg)  # 恢复原状态
    qc_reflect.h(reflect_qreg)

    # 重新初始化
    qc_reflect.append(init_circuit, reflect_qreg)

    return qc_reflect


def build_shift_operator(current_qubits, next_qubits):
    qc_shift = QuantumCircuit(current_qubits + next_qubits, name="Shift")
    for cq, nq in zip(current_qubits, next_qubits):
        qc_shift.swap(cq, nq)
    return qc_shift


# -------------------------------
# 主电路构建
# -------------------------------


def build_qpagerank_circuit(G, alpha=0.85, steps=2):
    """完整量子PageRank电路"""
    # 构建初始态电路并获取量子位列表
    init_qc, all_qubits = build_initial_state_circuit(G, alpha)
    n = int(np.ceil(np.log2(G.shape[0])))

    # 分离各寄存器量子位
    current_qubits = all_qubits[1 : 1 + n]
    next_qubits = all_qubits[1 + n :]

    # 构建反射算符
    reflect_circ = build_reflection_operator(init_qc)

    # 构建移位算符
    shift_circ = build_shift_operator(current_qubits, next_qubits)

    # 主电路
    main_qc = QuantumCircuit(*init_qc.qregs)
    main_qc.compose(init_qc, inplace=True)

    # 演化循环
    for _ in range(steps):
        # 反射操作（作用于所有量子位）
        main_qc.compose(reflect_circ, qubits=init_qc.qubits, inplace=True)
        # 移位操作
        main_qc.compose(
            shift_circ, qubits=current_qubits + next_qubits, inplace=True
        )  # 反射操作（作用于所有量子位）
        main_qc.compose(reflect_circ, qubits=init_qc.qubits, inplace=True)
        # 移位操作
        main_qc.compose(shift_circ, qubits=current_qubits + next_qubits, inplace=True)

    # 测量当前节点
    # main_qc.measure(current_qubits, main_qc.cregs[0])

    return main_qc


# -------------------------------
# 模拟与验证
# -------------------------------


def extract_partial_probability(prob_dict, qubit_indices):
    """
    提取指定比特的概率分布。
    :param prob_dict: 完整概率分布字典
    :param qubit_indices: 关心的比特索引列表（例如 [1, 2] 表示关心第1和第2个比特）
    :return: 提取的比特概率分布
    """
    partial_probs = {}
    for bitstr, prob in prob_dict.items():
        # 提取关心的比特状态
        state = "".join([bitstr[i] for i in qubit_indices])
        partial_probs[state] = partial_probs.get(state, 0) + prob
    return partial_probs


result_map = {}
delta_average = {}


def evolutionary(qc, previous_average=None):
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(qc_no_meas)

    # 模拟测量：计算状态向量中，当前节点寄存器（q_curr）的概率分布。
    prob_dict = sv.probabilities_dict()

    # 示例：关心 q_current（假设它是第2个寄存器，索引为 1 到 1+n-1）
    total_qubits = qc.qubits

    # qubit_indices = list(range(len(total_qubits)))[::-1]  # q_current 的索引范围
    qubit_indices = [6, 5, 4]  # q_current 的索引范围
    partial_probs = extract_partial_probability(prob_dict, qubit_indices)

    for state, prob in partial_probs.items():

        # 检查result_map中是否已经存在该节点
        if state in result_map:
            # 如果存在，将当前概率添加到列表中
            result_map[state].append(prob)
        else:
            # 如果不存在，初始化为包含当前概率的列表
            result_map[state] = [prob]
    # 计算每个state的平均概率
    next_average = {}

    for node, probs_list in result_map.items():
        # 计算平均值
        average = sum(probs_list) / len(probs_list)
        next_average[node] = average

    for node in next_average:
        if len(result_map[node]) == 1:
            delta = next_average[node]  # 如果是第一次计算，差值就是当前值
        else:
            delta = next_average[node] - previous_average[node]
        delta_average[node] = delta
    print(delta_average)
    return next_average


def quantum_pagerank(G, previous_average=None, alpha=0.85, steps=2):
    """量子PageRank模拟"""

    # 构建并运行电路
    qc = build_qpagerank_circuit(G, alpha, steps)

    # # 绘制电路图，保存为图片
    # fig = qc.draw(output="mpl", fold=-1, scale=0.8)
    # fig.savefig("q_pagerank_circuit1.png", bbox_inches="tight")
    previous_average = evolutionary(qc, previous_average)
    return previous_average


if __name__ == "__main__":
    # 示例网络（4节点）
    G = np.array(
        [
            [0.25, 0.45, 0.3, 0.4],
            [0.3, 0.15, 0.4, 0.4],
            [0.4, 0.25, 0.1, 0.1],
            [0.05, 0.15, 0.2, 0.1],
        ]
    )
    creatProbabilityMatrix = CreatProbabilityMatrix(
        r"D:\python\qpage-rank\Data\RoadSelect\stroke2.txt",
        r"D:\python\qpage-rank\Data\RoadSelect\node_data2.txt",
    )
    G = creatProbabilityMatrix.ProbabilityMatrix
    # quantum_pagerank(G, steps=1)
    # 运行量子PageRank
    previous_average = quantum_pagerank(G, steps=1)
    for i in range(10):
        quantum_pagerank(G, previous_average, steps=i + 2)

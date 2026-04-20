import numpy as np


class PageRankWeight:
    ProbabilityMatrix = []
    node_num = 0

    def __init__(self, ProbabilityMatrix, max_iterations, node_att):
        self.ProbabilityMatrix = ProbabilityMatrix
        self.node_num = ProbabilityMatrix.shape[0]
        self.max_iterations = max_iterations
        self.node_att = node_att

    def Google_matrix(self, alpha):
        O = np.ones(self.node_num)
        Google = alpha * self.ProbabilityMatrix + (1 - alpha) * O / self.node_num
        return Google

    def page_rank(self, alpha):
        Google = np.matrix(self.Google_matrix(alpha))
        start = [
            self.node_att[key]
            for key in sorted(self.node_att.keys(), key=lambda x: int(x))
        ]
        next_state = start
        t = 0
        diff_list = []
        while t < self.max_iterations:
            pre_state = next_state
            next_state = pre_state * Google
            t = t + 1
            diff = np.linalg.norm(pre_state - next_state)
            diff_list.append(diff)
        dic = []
        if self.max_iterations == 0:
            next_state = np.array(next_state)
        else:
            next_state = np.array(next_state)[0]
        for i in range(self.node_num):
            dic.append(np.around(next_state[i], 8))
        return dic, diff_list

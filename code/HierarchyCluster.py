import numpy as np

"""
层次聚类方法
(由距离矩阵开始计算)
类间距离定义方法：最大距离法(Single Link, SL)，平均距离法(Average Link, AL)
"""


class HC:
    def __init__(self, sim_mat, method):
        self.method = method
        self.sim_mat = sim_mat
        self._work()

    def AverageLinkage(self, A, B):
        """计算两个组合数据点中的每个数据点与其他所有数据点的距离
        将所有距离的均值作为两个组合数据点间的距离
        """
        total = 0.0
        for i in A:
            for j in B:
                total += self.sim_mat[i, j]
        ret = total / (np.shape(A)[0] * np.shape(B)[0])
        return ret

    def _work(self):

        pass
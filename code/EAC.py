import numpy as np
from cluster_method import *


def update_similar_mat(similar_mat, cluster_ind):
    """
    更新邻近度矩阵
    :param similar_mat: 原邻近度矩阵
    :param cluster_ind: 类群的样本序号列表
    :return:
    """
    if len(cluster_ind) <= 1:
        return similar_mat
    for ind, no_i in enumerate(cluster_ind):
        if no_i == cluster_ind[-1]:
            return similar_mat
        for no_j in cluster_ind[ind+1:]:
            similar_mat[no_i, no_j] += 1
            similar_mat[no_j, no_i] += 1


class EAC:
    def __init__(self, method=None, para_list=np.arange(3, 15, 1)):
        self.method = method
        self.para_list = para_list

    def single_fit(self, X, para):
        """
        单聚类方法
        :param X: 数据
        :param para: 参数
        :return:
        """
        cluster_result = self.method(X, k=para)
        return cluster_result

    def ensemble_fit(self, X):

        # 获取同质集成聚类结果并保存到字典
        result_dict = {}
        for k in self.para_list:
            result_dict.setdefault(k, self.single_fit(X, k))

        # 邻接度矩阵计算
        similar_mat = np.zeros((X.shape[0], X.shape[0]))
        for k, single_result in result_dict.items():
            k_num = list(set(single_result))
            for k_i in k_num:
                index_same_ki = np.where(single_result == k_i)[0]
                update_similar_mat(similar_mat, index_same_ki)
        print(similar_mat)






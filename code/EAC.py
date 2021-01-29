from cluster_method import *
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt


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

    def ensemble_fit(self, X, method='single', min_cluster=2, if_plot=0):
        """
        EAC 聚类集成
        :param X: 原数据
        :param method: 层次聚类连接方法
                single: 最短距离法
                weighted: 平均距离法
        :param min_cluster: 最小类群个数
        :param if_plot: 是否绘制层次聚类树状图
        :return: 聚类结果
        """
        # 获取同质集成聚类结果并保存到字典
        result_dict = {}
        for k in self.para_list:
            result_dict.setdefault(k, self.single_fit(X, k))

        # 邻近度矩阵计算
        similar_mat = np.zeros((X.shape[0], X.shape[0]))
        for k, single_result in result_dict.items():
            k_num = list(set(single_result))
            for k_i in k_num:
                index_same_ki = np.where(single_result == k_i)[0]
                update_similar_mat(similar_mat, index_same_ki)
        similar_mat += np.diag([len(self.para_list)+1] * X.shape[0])

        # 邻近度矩阵变换为距离矩阵
        distance_mat = 1 - similar_mat/(len(self.para_list)+1)

        # linkage层次聚类
        Z = linkage(distance_mat, method)

        # 挑选生存时间最长簇个数
        group_distance = Z[:-min_cluster, 2]  # 类间距离向量
        lifetime = group_distance[1:] - group_distance[:-1]  # 计算生存时间
        max_lifetime_index = np.argmax(lifetime)
        threshold = Z[max_lifetime_index, 2]  # 对应的类间距离

        # 聚类结果并画图
        cluster_result = fcluster(Z, threshold, 'distance')
        if if_plot:
            plt.figure(figsize=(5, 3))
            dendrogram(Z)
            plt.show()
        return cluster_result






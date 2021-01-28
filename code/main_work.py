import numpy as np
from cluster_method import *
from EAC import EAC


def load_data():
    ori_data = np.load(r'..\data\glass.npy')
    return ori_data


if __name__ == '__main__':
    # 载入数据
    data_all = load_data()
    label = data_all[:, 0]
    data = data_all[:, 1:]

    # 载入模型
    clustering = kmeans_cluster
    eac = EAC(clustering)
    eac.ensemble_fit(data)

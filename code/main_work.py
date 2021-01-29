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
    clustering = kmeans_cluster  # 选择K-means聚类方法
    para_list = np.arange(3, 15, 1)  # K-means的超参列表
    eac = EAC(clustering)

    # EAC集成聚类
    result = eac.ensemble_fit(data,
                              method='single',
                              if_plot=0)
    print(result)
    print(label)
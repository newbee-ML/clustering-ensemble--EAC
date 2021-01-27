import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN, KMeans, mean_shift
from sklearn.mixture import GaussianMixture
import os


# KDE-SA-DBSCAN的KDE-SA部分
def kde_sa(data, plot_yes=0, save_path=None):
    """
    基于核密度估计的DBSCAN超参数(Eps,minEps)的自适应估计问题


    :param data: 需要聚类的数据
                type:array
                size=(n_sample,n_feature)
    :param plot_yes: 是否画图
    :param save_path: 图片保存地址
    :return:
    """
    data = np.array(MinMaxScaler().fit_transform(data))
    distance_mat = np.zeros((data.shape[0], data.shape[0]))
    distance_mat_no_sorted = distance_mat
    # 计算距离矩阵
    for i in range(data.shape[0]):
        for j in range(i + 1, data.shape[0], 1):
            distance_mat_no_sorted[i, j] = np.sqrt(np.sum((data[j] - data[i]) ** 2))
            distance_mat_no_sorted[j, i] = distance_mat_no_sorted[i, j]
        distance_mat[i, :] = sorted(distance_mat_no_sorted[i, :])
    # 获得KNN-1矩阵
    knn_mat = distance_mat[:, 1:]
    for i in range(knn_mat.shape[1]):
        knn_mat[:, i] = sorted(knn_mat[:, i])
    bw = 0.05
    if plot_yes:

        # 梯状图
        plt.figure('Ladder chart 1')
        plt.rcParams['figure.figsize'] = (5.0, 5.0)
        plt.rcParams['savefig.dpi'] = 500  # 图片像素
        plt.rcParams['figure.dpi'] = 500  # 分辨率
        for k in range(2, 20, 1):  # range(3, 15, 1)
            plt.step(range(1, len(knn_mat[:, k]) + 1, 1), knn_mat[:, k], where="pre", lw=2,
                     label='minPts=' + str(k + 1))
        plt.xlabel('Sorted Samples')
        plt.ylabel('Distance')
        plt.legend(loc='upper left', fontsize=7)
        plt.savefig(os.path.join(save_path, 'Ladder chart 1.png'))

        # 核密度估计可视化
        plt.figure('Kernel Density Estimate Curve(Gaussian Kernel)')
        plt.rcParams['figure.figsize'] = (5.0, 5.0)
        plt.rcParams['savefig.dpi'] = 500  # 图片像素
        plt.rcParams['figure.dpi'] = 500  # 分辨率
        for k in range(2, 20, 1):
            data_sim = [[i] for i in knn_mat[:, k]]
            kde = KernelDensity(kernel='epanechnikov', bandwidth=bw).fit(data_sim)
            y = kde.score_samples(data_sim)
            y = np.exp(y)
            plt.plot(knn_mat[:, k], y, label='minPts=' + str(k + 1))
        plt.xlabel('Distance')
        plt.ylabel('Density')
        plt.legend(loc='upper right', fontsize=8)
        plt.savefig(os.path.join(save_path, 'Kernel Density Estimate Curve(Gaussian Kernel).png'))

        plt.figure('Kernel Density Estimate Curve(Gaussian Kernel, minPts=5)')
        plt.rcParams['figure.figsize'] = (5.0, 5.0)
        plt.rcParams['savefig.dpi'] = 500  # 图片像素
        plt.rcParams['figure.dpi'] = 500  # 分辨率
        sns.distplot(knn_mat[:, 4], axlabel="x", kde=True, kde_kws={"color": "g", "lw": 0},
                     rug=True, rug_kws={"color": "k"}, label='Density'
                     )
        sns.kdeplot(knn_mat[:, 4],
                    kernel='gau',
                    bw='scott',
                    label="KDE curve",
                    linewidth=2)
        plt.xlim((0, 0.15))
        plt.xlabel('Distance')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'Kernel Density Estimate Curve(Gaussian Kernel, minPts=5).png'))

    # 核密度估计
    k_max = 50  # 设置minEps的最大值
    eps_list = []
    for k in range(2, k_max, 1):
        data_sim = [[i] for i in knn_mat[:, k]]
        kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(data_sim)
        # [‘gaussian’|’tophat’|’epanechnikov’|’exponential’|’linear’|’cosine’]
        density_kde = np.exp(kde.score_samples(data_sim))
        decrease_density = density_kde[1:] - density_kde[:-1]
        decrease_distance = knn_mat[:, k][1:] - knn_mat[:, k][:-1]
        gradient = []
        for i in range(len(decrease_density)):
            if decrease_distance[i] == 0:
                gradient.append(0)
            else:
                gradient.append(decrease_density[i] / decrease_distance[i])
        gradient = np.array(gradient)
        index_max = np.where(gradient == np.min(gradient))[0]
        eps_list.append([k + 1, np.array(knn_mat[:, k])[index_max[-1]]])
    # 计算噪点个数
    noise_list = [[1, 0.001, 85], [2, 0.001, 18]]
    for info in eps_list:
        noise_wait2select = np.where(distance_mat[:, info[0]] > info[1])[0]  # 筛选非核心点，并获得索引
        noise_wait2select_set = set(noise_wait2select)
        noise_wait2select_set.add(0)
        noise_num = 0
        # 判断待选点的eps邻域是否有核心点
        for index in noise_wait2select:
            eps_index_set = set(np.where(distance_mat_no_sorted[index, :] <= info[1])[0])
            if eps_index_set <= noise_wait2select_set:
                noise_num += 1
        noise_list.append([info[0], info[1], noise_num])
    # 选择最好的minEps与EPS
    noise_array = np.array(noise_list)
    if plot_yes:
        plt.figure('noise')
        plt.rcParams['figure.figsize'] = (6.0, 6.0)
        plt.rcParams['savefig.dpi'] = 500  # 图片像素
        plt.rcParams['figure.dpi'] = 500  # 分辨率
        plt.step(noise_array[:, 0], noise_array[:, 2], where="pre", lw=2)
        plt.xlabel('minPts')
        plt.ylabel('noise number')
        plt.savefig(os.path.join(save_path, 'noise.png'))
        plt.close()
    diff_vec = np.array(abs(noise_array[:, 0] - noise_array[:, 2]))
    best_minEps_index = np.where(diff_vec == np.min(diff_vec))
    result = noise_array[best_minEps_index, [1, 0]][0]
    Eps = result[0]
    minPts = result[1]
    return Eps, minPts


"""cluster method"""


def DBSCAN_cluster(data, **kwargs):  # DBSCAN
    if kwargs is None:
        kwargs = {'eps': 0.1, 'minpts': 5}
    model = DBSCAN(kwargs['eps'], kwargs['minpts'])
    return model.fit_predict(np.array(MinMaxScaler().fit_transform(data[:, :2])))


def kmeans_cluster(data, **kwargs):  # K-means
    if kwargs is None:
        kwargs = {'k': 8}
    kmean = KMeans(n_clusters=kwargs['k'])
    return kmean.fit_predict(np.array(MinMaxScaler().fit_transform(data[:, :2])))


def mean_shift_cluster(data):  # mean_shift
    mean_shift_label = mean_shift(np.array(MinMaxScaler().fit_transform(data)))
    return list(mean_shift_label[-1])


def GMM_cluster(data, **kwargs):  # GMM
    gmm = GaussianMixture(n_components=kwargs['k']).fit(np.array(MinMaxScaler().fit_transform(data[:, :2])))
    return gmm.predict(np.array(MinMaxScaler().fit_transform(data[:, :2])))


def SA_DBSCAN_cluster(data, save_path):  # kde-sa-DBSCAN
    eps, minpts = kde_sa(data, plot_yes=1, save_path=save_path)
    return DBSCAN_cluster(data, eps=eps, minpts=minpts)

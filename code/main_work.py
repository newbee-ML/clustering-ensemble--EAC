import numpy as np


def load_data():
    ori_data = np.load(r'..\data\glass.npy')
    return ori_data


if __name__ == '__main__':
    # 载入数据
    data_all = load_data()
    label = data_all[:, 0]
    data = data_all[:, 1:]

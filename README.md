# EAC方法

### 复现论文基本信息

**原论文名称：**

Combining multiple clusterings using evidence accumulation

**原论文地址：**

https://ieeexplore.ieee.org/document/1432715



### 复现算法：

基于k-means聚类算法，在k=[10, 30]间选择N个超参数，将N个聚类结果通过EAC集成聚类方法进行集成：

**Step 1** N次K-means聚类获得N个聚类结果

**Step 2** 计算邻近度矩阵

**Step 3** 邻近度矩阵伪变换为距离矩阵

**Step 4** 层次聚类法对距离矩阵进行计算

**Step 5** 计算簇的生存周期，并挑选最大生存周期的簇结果作为聚类个数

**Step 6** 将上述层次聚类结果输出

Tips：与原论文不同，没有利用邻近度矩阵直接层次聚类，而是将其转化为伪距离矩阵进行计算。



### 调用方法：

```python
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
```


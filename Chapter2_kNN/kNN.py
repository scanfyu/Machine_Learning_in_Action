import numpy as np
import operator

def createDataSet():
    # 四组二维特征
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    # 四组特征的标签
    labels = ['A','A','B','B']
    return group, labels



# 2-1函数：kNN算法
def classify0(inX, dataSet, labels, k):
    """
    函数说明：kNN算法，分类器

    Parameters:
        inX - 用于分类的数据（测试集）
        dataSet - 用于训练的数据（训练集）（n*1维列向量）
        labels - 分类标准（n*1维列向量）
        k - kNN算法参数，选择距离最小的k个点
        
    Returns:
        sortedClassCount[0][0] - 分类结果
        
    Modify:
        2020-05-17
    """
    # numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    # 将inX重复dataSetSize次并排成一列
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 二维特征相减后平方（用diffMat的转置乘diffMat）
    sqDiffMat = diffMat**2
    # sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方，计算出距离
    distances = sqDistances**0.5
    # argsort函数返回的是distances值从小到大的--索引值
    sortedDistIndicies = distances.argsort()
    # 定义一个记录类别次数的字典
    classCount = {}
    # 选择距离最小的k个点
    for i in range(k):
        # 取出前k个元素的类别
        voteIlabel = labels[sortedDistIndicies[i]]
        # 字典的get()方法，返回指定键的值，如果值不在字典中返回0
        # 计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # python3中用items()替换python2中的iteritems()
    # key = operator.itemgetter(1)根据字典的值进行排序
    # key = operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典
    sortedClassCount = sorted(classCount.items(),
                              key = operator.itemgetter(1), reverse = True)
    # 返回次数最多的类别，即所要分类的类别
    return sortedClassCount[0][0]
